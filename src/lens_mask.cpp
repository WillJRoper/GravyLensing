/**
 * LensMask implementation
 *
 * This file is part of GravyLensing, a real-time gravitational lensing
 * simulation.
 *
 * GravyLensing is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GravyLensing is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GravyLensing. If not, see <http://www.gnu.org/licenses/>.
 */

#include "lens_mask.hpp"
#include "cam_feed.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <iostream>
#include <opencv2/imgproc.hpp> // for cv::cvtColor, cv::compare, etc.
#include <torch/script.h>      // for torch::jit::script::Module
#include <torch/torch.h>       // for torch::cuda::is_available, tensor ops

// -----------------------
// Constructor / Destructor
// -----------------------
LensMask::LensMask(CameraFeed *camFeed, float softening, int padFactor,
                   float strength, const std::string &modelPath, int maskScale)
    : camFeed_(camFeed),
      // Set the dimensions of the lens mask
      width_(camFeed->width_), height_(camFeed->height_),
      // Set the parameters for the lensing effect
      softening_(softening), strength_(strength), maskScale_(maskScale),
      // The torch model path
      modelPath_(modelPath),
      // The padding factor for the FFT
      padFactor_(padFactor),
      // Pick device at construction time:
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

  // Ensure we always have a valid mask and lensed image of the right size
  latestMask_ = cv::Mat::zeros(height_, width_, CV_8UC1);
  latestLensed_ = cv::Mat::zeros(height_, width_, CV_8UC3);

  // Update the Geometry (we call this function explicitly because it
  // also handles all the FFTW allocations and builds the kernels)
  updateGeometry(width_, height_);

  // Load TorchScript segmentation model onto device_
  try {
    segmentModel_ = torch::jit::load(modelPath, device_);
    segmentModel_.eval();
  } catch (const c10::Error &e) {
    std::cerr << "Error loading segmentation model:\n" << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Pre-allocate the CHW float tensor on the right device
  inputTensor_ = torch::empty(
      {1, 3, fastH_, fastW_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
}

void LensMask::allocateFFTWKernels() {
  // Allocate the FFTW buffers for the kernels
  kernelX_ = (float *)fftwf_malloc(sizeof(float) * padHeight_ * padWidth_);
  kernelY_ = (float *)fftwf_malloc(sizeof(float) * padHeight_ * padWidth_);
  Kx_ft_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * padHeight_ *
                                         (padWidth_ / 2 + 1));
  Ky_ft_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * padHeight_ *
                                         (padWidth_ / 2 + 1));

  // Create FFTW plans (real-to-complex 2D)
  planKx_ = fftwf_plan_dft_r2c_2d(padHeight_, padWidth_, kernelX_, Kx_ft_,
                                  FFTW_MEASURE);
  planKy_ = fftwf_plan_dft_r2c_2d(padHeight_, padWidth_, kernelY_, Ky_ft_,
                                  FFTW_MEASURE);
}

void LensMask::freeFFTWKernels() {
  if (planKx_)
    fftwf_destroy_plan(planKx_);
  if (planKy_)
    fftwf_destroy_plan(planKy_);
  if (kernelX_)
    fftwf_free(kernelX_);
  if (kernelY_)
    fftwf_free(kernelY_);
  if (Kx_ft_)
    fftwf_free(Kx_ft_);
  if (Ky_ft_)
    fftwf_free(Ky_ft_);
}

void LensMask::allocateFFTWDeflections() {
  maskBuf_ = (float *)fftwf_malloc(sizeof(float) * padHeight_ * padWidth_);
  maskFT_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * padHeight_ *
                                          (padWidth_ / 2 + 1));
  defXFT_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * padHeight_ *
                                          (padWidth_ / 2 + 1));
  defYFT_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * padHeight_ *
                                          (padWidth_ / 2 + 1));
  defXBuf_ = (float *)fftwf_malloc(sizeof(float) * padHeight_ * padWidth_);
  defYBuf_ = (float *)fftwf_malloc(sizeof(float) * padHeight_ * padWidth_);

  planMask_ = fftwf_plan_dft_r2c_2d(padHeight_, padWidth_, maskBuf_, maskFT_,
                                    FFTW_MEASURE);
  planDefX_ = fftwf_plan_dft_c2r_2d(padHeight_, padWidth_,
                                    defXFT_,  // complex input
                                    defXBuf_, // real output
                                    FFTW_MEASURE);

  planDefY_ = fftwf_plan_dft_c2r_2d(padHeight_, padWidth_,
                                    defYFT_,  // complex input
                                    defYBuf_, // real output
                                    FFTW_MEASURE);
}

void LensMask::freeFFTWDeflections() {
  if (maskBuf_)
    fftwf_free(maskBuf_);
  if (maskFT_)
    fftwf_free(maskFT_);
  if (defXFT_)
    fftwf_free(defXFT_);
  if (defYFT_)
    fftwf_free(defYFT_);
  if (defXBuf_)
    fftwf_free(defXBuf_);
  if (defYBuf_)
    fftwf_free(defYBuf_);
  if (planMask_)
    fftwf_destroy_plan(planMask_);
  if (planDefX_)
    fftwf_destroy_plan(planDefX_);
  if (planDefY_)
    fftwf_destroy_plan(planDefY_);
}

LensMask::~LensMask() {
  // Free the FFTW plans and buffers
  freeFFTWKernels();

  // Free the deflection kernels
  freeFFTWDeflections();

  // Free the input tensor
  inputTensor_.reset();
}

// -----------------------
// Person Segmentation
// -----------------------
void LensMask::detectPersonMask() {
  // 0) Bail if no new frame
  if (!camFeed_->newFrameReady_)
    return;

  // 1) Grab the latest frame under lock
  cv::Mat frame;
  {
    std::lock_guard lk(camFeed_->frameMutex_);
    if (camFeed_->latestFrame_.empty())
      return;
    frame = camFeed_->latestFrame_.clone();
    camFeed_->newFrameReady_ = false;
  }

  // 2) Downsample & convert to RGB
  cv::Mat smallFrame, rgb;
  cv::resize(frame, smallFrame, cv::Size(fastW_, fastH_), 0, 0,
             cv::INTER_LINEAR);
  cv::cvtColor(smallFrame, rgb, cv::COLOR_BGR2RGB);

  // 3) Copy into pre-allocated tensor and normalize to [0,1]
  float *tptr = inputTensor_.data_ptr<float>();
  for (int y = 0; y < fastH_; ++y) {
    const cv::Vec3b *row = rgb.ptr<cv::Vec3b>(y);
    for (int x = 0; x < fastW_; ++x) {
      tptr[0 * fastH_ * fastW_ + y * fastW_ + x] = row[x][0] / 255.f;
      tptr[1 * fastH_ * fastW_ + y * fastW_ + x] = row[x][1] / 255.f;
      tptr[2 * fastH_ * fastW_ + y * fastW_ + x] = row[x][2] / 255.f;
    }
  }
  auto tt = inputTensor_;
  tt[0][0].sub_(0.485f).div_(0.229f);
  tt[0][1].sub_(0.456f).div_(0.224f);
  tt[0][2].sub_(0.406f).div_(0.225f);

  // 4) Run the model
  torch::NoGradGuard no_grad;
  auto out_iv = segmentModel_.forward({inputTensor_});
  torch::Tensor logits;
  if (out_iv.isTensor())
    logits = out_iv.toTensor();
  else if (out_iv.isTuple())
    logits = out_iv.toTuple()->elements()[0].toTensor();
  else if (out_iv.isGenericDict())
    logits = out_iv.toGenericDict().at("out").toTensor();
  else {
    std::cerr << "Unexpected IValue from segmentation\n";
    return;
  }

  // 5) Convert logits → class map
  auto cm = logits.to(torch::kCPU)
                .squeeze(0) // [C,fastH_,fastW_] → [fastH_,fastW_]
                .argmax(0)  // pick class
                .to(torch::kUInt8);

  // 6) Wrap in cv::Mat and threshold for “person” (class 15)
  cv::Mat fastMask(fastH_, fastW_, CV_8UC1, cm.data_ptr<uint8_t>());
  cv::Mat binMask;
  cv::compare(fastMask, /*classID=*/15, binMask, cv::CMP_EQ);

  // 7) Upsample to full resolution and publish under lock
  {
    std::lock_guard lk(maskMutex_);
    newMaskReady_ = true;
    cv::resize(binMask, latestMask_, cv::Size(width_, height_), 0, 0,
               cv::INTER_NEAREST);
  }
}

void LensMask::shadeMask(cv::Mat &frame) {
  cv::Mat maskCopy;
  {
    // Grab the latest mask under lock
    std::lock_guard lk(maskMutex_);
    if (latestMask_.empty()) {
      // No mask computed yet → nothing to shade
      return;
    }
    maskCopy = latestMask_;
  }

  // If mask and frame differ in size, upsample nearest-neighbor:
  if (maskCopy.size() != frame.size()) {
    cv::resize(maskCopy, maskCopy, frame.size(), 0, 0, cv::INTER_NEAREST);
  }

  // Ensure mask is 8-bit single channel:
  if (maskCopy.type() != CV_8UC1) {
    maskCopy.convertTo(maskCopy, CV_8UC1);
  }

  // Ensure frame is 3-channel 8-bit BGR:
  if (frame.type() != CV_8UC3) {
    if (frame.channels() == 1) {
      frame.convertTo(frame, CV_8U);
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    } else {
      frame.convertTo(frame, CV_8UC3);
    }
  }

  // Paint red where mask==1
  for (int y = 0; y < frame.rows; ++y) {
    const uchar *mrow = maskCopy.ptr<uchar>(y);
    cv::Vec3b *prow = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < frame.cols; ++x) {
      if (mrow[x]) {
        prow[x] = cv::Vec3b(0, 0, 255);
      }
    }
  }
}

// Build the real-space deflection kernels and FFT them
void LensMask::buildKernels() {
  const int H = padHeight_;
  const int W = padWidth_;
  const int N = H * W;
  const int Wc = W / 2 + 1;
  const float norm = 1.0f / (static_cast<float>(H) * W);

  // Physical (periodic) domain in pixel units
  const float dx = 1.0f;
  const float dy = 1.0f;
  const float physW = W * dx;
  const float physH = H * dy;
  const float halfPhysW = physW * 0.5f;
  const float halfPhysH = physH * 0.5f;

  // Cutoff radius (pixels), taper fraction
  const float padExtraW = static_cast<float>(padWidth_ - width_);
  const float padExtraH = static_cast<float>(padHeight_ - height_);
  const float rcutoff = std::min(padExtraW, padExtraH) * 0.5f;
  const float CUTOFF = 0.2f;
  const float eps2 = softening_ * softening_;

  // Build the real-space kernel with periodic wrap-around and cosine taper
  for (int j = 0; j < H; ++j) {
    for (int i = 0; i < W; ++i) {
      // Map into periodic domain
      float x = (i + 0.5f) * dx;
      if (x > halfPhysW)
        x -= physW;
      float y = (j + 0.5f) * dy;
      if (y > halfPhysH)
        y -= physH;

      float r = std::sqrt(x * x + y * y);
      float r2 = r * r + eps2;

      // Smooth cutoff window
      float fac;
      if (r > rcutoff) {
        fac = 0.0f;
      } else if (r > CUTOFF * rcutoff) {
        float f = (r - CUTOFF * rcutoff) / ((1.0f - CUTOFF) * rcutoff);
        fac = 0.5f * (std::cos(M_PI * f) + 1.0f);
      } else {
        fac = 1.0f;
      }

      // Base kernel value (includes 1/(H*W) normalization)
      float base = 1.0f / (static_cast<float>(M_PI) * r2) * norm;
      kernelX_[j * W + i] = x * base * fac;
      kernelY_[j * W + i] = y * base * fac;
    }
  }

  // Execute the plans to fill Kx_ft_ and Ky_ft_
  fftwf_execute(planKx_);
  fftwf_execute(planKy_);
}

// -----------------------
// Apply Lensing
// -----------------------
void LensMask::applyLensing(const cv::Mat &background) {

  // If we have no new mask to process just pass through the background
  if (!newMaskReady_) {
    return;
  }

  // Get a local copy of the mask under lock and then flag we are done with it
  cv::Mat maskCopy;
  {
    std::lock_guard lk(maskMutex_);
    if (latestMask_.empty()) {
      return;
    }
    latestMask_.copyTo(maskCopy);
    newMaskReady_ = false;
  }

  // Lazy shorthands
  const int H = height_, W = width_;
  const int pH = padHeight_, pW = padWidth_;
  const int pWC = pW / 2 + 1;

  // Fill maskBuf_ with padded mask (float 0/1)
  {
    // 1) convert maskCopy (uchar 0/255) → float 0/1
    cv::Mat floatMask;
    maskCopy.convertTo(floatMask, CV_32F, 1.0f / 255.0f);

    // 2) compute border sizes
    const int top = (pH - H) / 2;
    const int left = (pW - W) / 2;
    const int bottom = pH - H - top;
    const int right = pW - W - left;

    // 3) pad with zero (no periodic/reflect artifacts)
    cv::Mat paddedMask;
    cv::copyMakeBorder(floatMask, paddedMask, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0.0f));

    // 4) copy into your raw float* buffer in one go
    std::memcpy(maskBuf_, paddedMask.ptr<float>(),
                size_t(pH) * size_t(pW) * sizeof(float));
  }

  // FFT the mask → maskFT_
  {
    fftwf_execute(planMask_); // maskBuf_ → maskFT_
  }

  // Multiply in Fourier space: defXFT_ = maskFT_ * Kx_ft_, same for Y
  {
    for (int idx = 0, end = pH * pWC; idx < end; ++idx) {
      float a = maskFT_[idx][0], b = maskFT_[idx][1];
      float c = Kx_ft_[idx][0], d = Kx_ft_[idx][1];
      defXFT_[idx][0] = a * c - b * d;
      defXFT_[idx][1] = a * d + b * c;
      c = Ky_ft_[idx][0];
      d = Ky_ft_[idx][1];
      defYFT_[idx][0] = a * c - b * d;
      defYFT_[idx][1] = a * d + b * c;
    }
  }

  // Inverse FFT → defXBuf_, defYBuf_
  {
    fftwf_execute(planDefX_);
    fftwf_execute(planDefY_);
  }

  // Build remap arrays and crop to [H×W]
  cv::Mat mapX(H, W, CV_32FC1), mapY(H, W, CV_32FC1);
  {
    int offY = (pH - H) / 2, offX = (pW - W) / 2;
    for (int y = 0; y < H; ++y) {
      float *mpx = mapX.ptr<float>(y);
      float *mpy = mapY.ptr<float>(y);
      const uchar *msk = maskCopy.ptr<uchar>(y);
      for (int x = 0; x < W; ++x) {
        float dx = defXBuf_[(y + offY) * pW + (x + offX)] * strength_;
        float dy = defYBuf_[(y + offY) * pW + (x + offX)] * strength_;
        // if (msk[x]) {
        //   dx = dy = 0.f;
        // }
        mpx[x] = std::clamp(x + dx, 0.0f, float(W - 1));
        mpy[x] = std::clamp(y + dy, 0.0f, float(H - 1));
      }
    }
  }

  // Perform the remap to get to our actual output
  {
    std::lock_guard lk(lensedMutex_);
    cv::remap(background, latestLensed_, mapX, mapY, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0) // black border
    );

    // Flag that we have a new lensed image
    newLensedImageReady_ = true;
  }
}

void LensMask::startAsyncSegmentation() {
  stopWorker_ = false;
  workerThread_ = std::thread(&LensMask::segmentationLoop, this);
}

void LensMask::stopAsyncSegmentation() {
  stopWorker_ = true;
  if (workerThread_.joinable())
    workerThread_.join();
}

void LensMask::segmentationLoop() {
  int nframes = 0;
  while (!stopWorker_) {
    cv::Mat frameCopy;
    {
      std::lock_guard lk(camFeed_->frameMutex_);
      camFeed_->latestFrame_.copyTo(frameCopy);
    }

    // Run your downsample+infer+upsample pipeline on frameCopy:
    cv::Mat mask;
    detectPersonMask();

    // Small pause to yield CPU
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
}

/**
 * @brief Update the geometry of the lens.
 *
 * This will also free and reallocate all the FFTW plans and buffers, to
 * account for the new dimensions.
 *
 * @param width  New width of the lens
 * @param height  New height of the lens
 */
void LensMask::updateGeometry(int width, int height) {
  // Update the lens geometry
  width_ = width;
  height_ = height;

  // Update the padded dimensions
  padWidth_ = width_ * padFactor_;
  padHeight_ = height_ * padFactor_;

  // Update the fast dimensions
  fastW_ = width_ / maskScale_;
  fastH_ = height_ / maskScale_;

  // We need to reallocate the FFTW plans and buffers for the kernels and then
  // rebuild everything

  // Free all the FFTW plans and buffers
  freeFFTWKernels();
  freeFFTWDeflections();

  // And reallocate the FFTW plans and buffers for the kernels
  allocateFFTWKernels();
  allocateFFTWDeflections();

  // Rebuild the kernels now we've redone all our allocations
  buildKernels();

  // Rebuild the segmentation model input tensor
  inputTensor_ = torch::empty(
      {1, 3, fastH_, fastW_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  // Rebuild the segmentation model
  try {
    segmentModel_ = torch::jit::load(modelPath_, device_);
    segmentModel_.eval();
  } catch (const c10::Error &e) {
    std::cerr << "Error loading segmentation model:\n" << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Rebuild the mask and lensed images
  latestMask_ = cv::Mat::zeros(height_, width_, CV_8UC1);
  latestLensed_ = cv::Mat::zeros(height_, width_, CV_8UC3);

  // Update the camera feed
  camFeed_->updateGeometry(width_, height_);
}
