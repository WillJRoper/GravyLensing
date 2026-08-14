/**
 * @file lensing_worker.cpp
 *
 * This defines the worker class used to calculate the lensing effect. A
 * new lens is calculated for every new mask.
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

// Standard includes
#include <chrono>
#include <cstring>

// Local includes
#include "lensing_worker.hpp"
#include "perf_log.hpp"
#include "processing_geometry.hpp"

#ifdef USE_MPS
#include "metal_helper.h"

static const char *kLensingShader = R"(
#include <metal_stdlib>
using namespace metal;

struct RemapParams {
    int width;
    int height;
    int padW;
    int offX;
    int offY;
    float strength;
    bool distortInside;
};

kernel void buildRemapMaps(
    device const float *defX        [[buffer(0)]],
    device const float *defY        [[buffer(1)]],
    device const uchar  *mask       [[buffer(2)]],
    device float *mapX              [[buffer(3)]],
    device float *mapY              [[buffer(4)]],
    constant RemapParams &params    [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.width * params.height) return;
    int y = idx / params.width;
    int x = idx - y * params.width;
    
    if (!params.distortInside && mask[idx] > 0) {
        mapX[idx] = float(x);
        mapY[idx] = float(y);
        return;
    }
    
    int pidx = (y + params.offY) * params.padW + (x + params.offX);
    float dx = defX[pidx] * params.strength;
    float dy = defY[pidx] * params.strength;
    float xx = float(x) + dx;
    float yy = float(y) + dy;
    float maxX = float(params.width - 1);
    float maxY = float(params.height - 1);
    mapX[idx] = xx < 0.0f ? 0.0f : (xx > maxX ? maxX : xx);
    mapY[idx] = yy < 0.0f ? 0.0f : (yy > maxY ? maxY : yy);
}

inline float sampleChannel(device const uchar *image, int width, int height,
                           float x, float y, int channel) {
    float xx = clamp(x, 0.0f, float(width - 1));
    float yy = clamp(y, 0.0f, float(height - 1));
    int x0 = int(floor(xx));
    int y0 = int(floor(yy));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    float wx = xx - float(x0);
    float wy = yy - float(y0);

    int idx00 = (y0 * width + x0) * 3 + channel;
    int idx10 = (y0 * width + x1) * 3 + channel;
    int idx01 = (y1 * width + x0) * 3 + channel;
    int idx11 = (y1 * width + x1) * 3 + channel;

    float top = mix(float(image[idx00]), float(image[idx10]), wx);
    float bottom = mix(float(image[idx01]), float(image[idx11]), wx);
    return mix(top, bottom, wy);
}

kernel void renderLensedImage(
    device const float *defX        [[buffer(0)]],
    device const float *defY        [[buffer(1)]],
    device const uchar *mask        [[buffer(2)]],
    device const uchar *background  [[buffer(3)]],
    device uchar *output            [[buffer(4)]],
    constant RemapParams &params    [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.width * params.height) return;
    int y = idx / params.width;
    int x = idx - y * params.width;
    int outBase = idx * 3;

    float xx = float(x);
    float yy = float(y);
    if (params.distortInside || mask[idx] == 0) {
        int pidx = (y + params.offY) * params.padW + (x + params.offX);
        xx += defX[pidx] * params.strength;
        yy += defY[pidx] * params.strength;
    }

    output[outBase + 0] = uchar(round(sampleChannel(background, params.width,
                                                    params.height, xx, yy, 0)));
    output[outBase + 1] = uchar(round(sampleChannel(background, params.width,
                                                    params.height, xx, yy, 1)));
    output[outBase + 2] = uchar(round(sampleChannel(background, params.width,
                                                    params.height, xx, yy, 2)));
}
)";

static metal::Pipeline *gRemapPipeline = nullptr;
static metal::Buffers *gRemapBufs = nullptr;
static int gRemapBufW = 0, gRemapBufH = 0, gRemapBufPW = 0;
static metal::Pipeline *gRenderPipeline = nullptr;
static metal::Buffers *gRenderBufs = nullptr;
static int gRenderBufW = 0, gRenderBufH = 0, gRenderBufPW = 0;
static bool gRenderBackgroundDirty = true;

static void ensureRemapBuffers(int W, int H, int pW, int pH) {
  if (gRemapBufW == W && gRemapBufH == H && gRemapBufPW == pW)
    return;

  metal::releaseBuffers(gRemapBufs);
  gRemapBufs = nullptr;

  int N2 = pH * pW;
  int HW = H * W;
  const unsigned long lens[] = {
      (unsigned long)(N2 * sizeof(float)),    // defX
      (unsigned long)(N2 * sizeof(float)),    // defY
      (unsigned long)(HW * sizeof(uchar)),    // mask
      (unsigned long)(HW * sizeof(float)),    // mapX
      (unsigned long)(HW * sizeof(float)),    // mapY
      64,                                      // params struct (padded)
  };
  gRemapBufs = metal::createBuffers(lens, 6);
  gRemapBufW = W;
  gRemapBufH = H;
  gRemapBufPW = pW;
}

static void ensureRenderBuffers(int W, int H, int pW, int pH) {
  if (gRenderBufW == W && gRenderBufH == H && gRenderBufPW == pW)
    return;

  metal::releaseBuffers(gRenderBufs);
  gRenderBufs = nullptr;

  int N2 = pH * pW;
  int HW = H * W;
  const unsigned long lens[] = {
      (unsigned long)(N2 * sizeof(float)),    // defX
      (unsigned long)(N2 * sizeof(float)),    // defY
      (unsigned long)(HW * sizeof(uchar)),    // mask
      (unsigned long)(HW * 3 * sizeof(uchar)),// background
      (unsigned long)(HW * 3 * sizeof(uchar)),// output
      64,                                     // params struct
  };
  gRenderBufs = metal::createBuffers(lens, 6);
  gRenderBufW = W;
  gRenderBufH = H;
  gRenderBufPW = pW;
  gRenderBackgroundDirty = true;
}
#endif

namespace {

void fillPaddedMaskReflect101(const cv::Mat &mask, float *dst, int padHeight,
                              int padWidth, int nthreads) {
  const int srcHeight = mask.rows;
  const int srcWidth = mask.cols;
  const int top = (padHeight - srcHeight) / 2;
  const int left = (padWidth - srcWidth) / 2;
  constexpr float kInv255 = 1.0f / 255.0f;

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int y = 0; y < padHeight; ++y) {
    const int srcY = cv::borderInterpolate(y - top, srcHeight,
                                           cv::BORDER_REFLECT_101);
    const uchar *srcRow = mask.ptr<uchar>(srcY);
    float *dstRow = dst + static_cast<size_t>(y) * padWidth;
    for (int x = 0; x < padWidth; ++x) {
      const int srcX = cv::borderInterpolate(x - left, srcWidth,
                                             cv::BORDER_REFLECT_101);
      dstRow[x] = static_cast<float>(srcRow[srcX]) * kInv255;
    }
  }
}

void fillPaddedMassReflect101(const cv::Mat &mass, float *dst, int padHeight,
                              int padWidth, int nthreads) {
  const int srcHeight = mass.rows;
  const int srcWidth = mass.cols;
  const int top = (padHeight - srcHeight) / 2;
  const int left = (padWidth - srcWidth) / 2;

#pragma omp parallel for num_threads(nthreads) schedule(static)
  for (int y = 0; y < padHeight; ++y) {
    const int srcY = cv::borderInterpolate(y - top, srcHeight,
                                           cv::BORDER_REFLECT_101);
    const float *srcRow = mass.ptr<float>(srcY);
    float *dstRow = dst + static_cast<size_t>(y) * padWidth;
    for (int x = 0; x < padWidth; ++x) {
      const int srcX = cv::borderInterpolate(x - left, srcWidth,
                                             cv::BORDER_REFLECT_101);
      dstRow[x] = srcRow[srcX];
    }
  }
}

} // namespace

/**
 * @brief Constructor for the LensingWorker class.
 *
 * @param strength The strength of the lensing effect.
 * @param softening The softening parameter for the lensing effect.
 * @param padFactor The padding factor for the lensing effect.
 * @param nthreads The number of threads to use for processing.
 */
LensingWorker::LensingWorker(float strength, float softening, int padFactor,
                              int nthreads, float lowerRes, bool distortInside,
                             float massBlurSigma)
    : strength_(strength), softening_(softening), padFactor_(padFactor),
      nthreads_(nthreads), lowerRes_(lowerRes), distortInside_(distortInside),
      massBlurSigma_(massBlurSigma) {

  std::cout << "[LensingWorker] Initializing lensing worker...\n";
  std::cout << "[LensingWorker] Strength: " << strength_ << "\n";
  std::cout << "[LensingWorker] Softening: " << softening_ << "\n";
  std::cout << "[LensingWorker] Padding factor: " << padFactor_ << "\n";
  std::cout
      << "[LensingWorker] Number of threads (excluding those taken by Qt): "
      << nthreads_ << "\n";
  std::cout << "[LensingWorker] Mass blur sigma: " << massBlurSigma_ << "\n";
}

void LensingWorker::submitMask(const cv::Mat &mask) {
  if (mask.empty()) {
    return;
  }

  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lock(pendingMaskMutex_);
    pendingMask_ = mask;
    if (!pendingMaskDrainScheduled_) {
      pendingMaskDrainScheduled_ = true;
      shouldSchedule = true;
    }
  }

  if (shouldSchedule) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingMask(); },
                              Qt::QueuedConnection);
  }
}

void LensingWorker::drainPendingMask() {
  cv::Mat mask;
  {
    std::lock_guard<std::mutex> lock(pendingMaskMutex_);
    if (pendingMask_.empty()) {
      pendingMaskDrainScheduled_ = false;
      return;
    }
    mask = std::move(pendingMask_);
    pendingMask_.release();
  }

  onMask(mask);

  bool shouldContinue = false;
  {
    std::lock_guard<std::mutex> lock(pendingMaskMutex_);
    if (pendingMask_.empty()) {
      pendingMaskDrainScheduled_ = false;
    } else {
      shouldContinue = true;
    }
  }

  if (shouldContinue) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingMask(); },
                              Qt::QueuedConnection);
  }
}

void LensingWorker::setStrength(float strength) {
  strength_ = strength;
  std::cout << "[LensingWorker] Strength updated to " << strength_ << "\n";
}

void LensingWorker::setDistortInside(bool distortInside) {
  distortInside_ = distortInside;
  std::cout << "[LensingWorker] DistortInside updated to "
            << (distortInside_ ? "true" : "false") << "\n";
}

/**
 * @brief Allocate FFTW kernels for the lensing effect.
 *
 * This function allocates the FFTW buffers and creates the FFTW plans for the
 * lensing effect.
 */
void LensingWorker::allocateFFTWKernels() {

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

/**
 * @brief Free the FFTW kernels and plans.
 *
 * This function frees the FFTW buffers and destroys the FFTW plans for the
 * lensing effect.
 */
void LensingWorker::freeFFTWKernels() {

  // Destroy the FFTW plans
  if (planKx_)
    fftwf_destroy_plan(planKx_);
  if (planKy_)
    fftwf_destroy_plan(planKy_);

  // Free the FFTW buffers
  if (kernelX_)
    fftwf_free(kernelX_);
  if (kernelY_)
    fftwf_free(kernelY_);
  if (Kx_ft_)
    fftwf_free(Kx_ft_);
  if (Ky_ft_)
    fftwf_free(Ky_ft_);
}

/**
 * @brief Allocate FFTW buffers for the deflections.
 *
 * This function allocates the FFTW buffers and creates the FFTW plans for the
 * deflections.
 */
void LensingWorker::allocateFFTWDeflections() {
  const int pH = padHeight_;
  const int pW = padWidth_;
  const int pWC = pW / 2 + 1;

  // mask as before
  maskBuf_ = (float *)fftwf_malloc(sizeof(float) * pH * pW);
  maskFT_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * pH * pWC);

  // batched deflection buffers
  defFT_ = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * 2 * pH * pWC);
  defBuf_ = (float *)fftwf_malloc(sizeof(float) * 2 * pH * pW);

  // forward of mask
  planMask_ = fftwf_plan_dft_r2c_2d(pH, pW, maskBuf_, maskFT_, FFTW_MEASURE);

  // build a single batched inverse‐FFT plan for 2 transforms
  int rank = 2;
  int n[2] = {pH, pW};
  int howmany = 2;
  int idist = pH * pWC; // distance between blocks in input
  int odist = pH * pW;  // distance between blocks in output

  planDef_ = fftwf_plan_many_dft_c2r(
      rank, n, howmany, defFT_, /*in*/ nullptr, /*in_strides*/ 1, idist,
      defBuf_, /*out*/ nullptr, /*out_strides*/ 1, odist, FFTW_MEASURE);
}

/**
 * @brief Free the FFTW buffers and plans for the deflections.
 *
 * This function frees the FFTW buffers and destroys the FFTW plans for the
 * deflections.
 */
void LensingWorker::freeFFTWDeflections() {

  // Destroy the FFTW plans
  if (planMask_)
    fftwf_destroy_plan(planMask_);
  if (planDef_)
    fftwf_destroy_plan(planDef_);

  // Free the FFTW buffers
  if (maskBuf_)
    fftwf_free(maskBuf_);
  if (maskFT_)
    fftwf_free(maskFT_);
  if (defFT_)
    fftwf_free(defFT_);
  if (defBuf_)
    fftwf_free(defBuf_);
}

LensingWorker::~LensingWorker() {
  // Free the FFTW plans and buffers
  freeFFTWKernels();

  // Free the deflection kernels
  freeFFTWDeflections();
}

/**
 * @brief Build the kernels for the lensing effect.
 *
 * This function builds the kernels for the lensing effect using FFTW.
 */
void LensingWorker::buildKernels() {

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

/**
 * @brief Apply the lensing effect to the current background.
 *
 * This function applies the lensing effect to the current background image
 * using the provided mask.
 *
 * @param mask The mask to apply the lensing effect.
 */
void LensingWorker::applyLensing(const cv::Mat &mask) {
  static thread_local PerfLog perfPad("lensing-pad", 60);
  static thread_local PerfLog perfForwardFft("lensing-forward-fft", 60);
  static thread_local PerfLog perfSpectralMul("lensing-spectral-mul", 60);
  static thread_local PerfLog perfInverseFft("lensing-inverse-fft", 60);
  static thread_local PerfLog perfMapBuild("lensing-map-build", 60);
  static thread_local PerfLog perfRemap("lensing-remap", 60);

  // Get some helpful constants shorthands
  const int H = height_;
  const int W = width_;
  const int pH = padHeight_;
  const int pW = padWidth_;
  const int pWC = pW / 2 + 1;
  const int N1 = pH * pWC; // complex bins per deflection
  const int N2 = pH * pW;  // real samples per deflection

  const auto tPad0 = std::chrono::steady_clock::now();
  mask.convertTo(softMassMask_, CV_32F, 1.0f / 255.0f);
  if (massBlurSigma_ > 0.0f) {
    cv::GaussianBlur(softMassMask_, softMassMask_, cv::Size(0, 0),
                     massBlurSigma_, massBlurSigma_, cv::BORDER_REPLICATE);
  }
  fillPaddedMassReflect101(softMassMask_, maskBuf_, pH, pW, nthreads_);
  const auto tPad1 = std::chrono::steady_clock::now();
  perfPad.addSample(
      std::chrono::duration<double, std::milli>(tPad1 - tPad0).count());

  // Forward FFT of mask
  const auto tFft0 = std::chrono::steady_clock::now();
  fftwf_execute(planMask_);
  const auto tFft1 = std::chrono::steady_clock::now();
  perfForwardFft.addSample(
      std::chrono::duration<double, std::milli>(tFft1 - tFft0).count());

  // Multiply in Fourier space into defFT_ blocks
  const auto tMul0 = std::chrono::steady_clock::now();
  {
#pragma omp parallel for simd num_threads(nthreads_)                           \
    aligned(maskFT_, Kx_ft_, Ky_ft_, defFT_ : 64) schedule(static)
    for (int i = 0; i < N1; ++i) {
      // load once
      float ar = maskFT_[i][0];
      float ai = maskFT_[i][1];
      float kxr = Kx_ft_[i][0];
      float kxi = Kx_ft_[i][1];
      float kyr = Ky_ft_[i][0];
      float kyi = Ky_ft_[i][1];

      // X block
      defFT_[i][0] = ar * kxr - ai * kxi;
      defFT_[i][1] = ar * kxi + ai * kxr;
      // Y block
      defFT_[N1 + i][0] = ar * kyr - ai * kyi;
      defFT_[N1 + i][1] = ar * kyi + ai * kyr;
    }
  }
  const auto tMul1 = std::chrono::steady_clock::now();
  perfSpectralMul.addSample(
      std::chrono::duration<double, std::milli>(tMul1 - tMul0).count());

  // Batched inverse FFT → defBuf_ contains [X; Y]
  const auto tInv0 = std::chrono::steady_clock::now();
  fftwf_execute(planDef_);
  const auto tInv1 = std::chrono::steady_clock::now();
  perfInverseFft.addSample(
      std::chrono::duration<double, std::milli>(tInv1 - tInv0).count());

  // Build remap maps from defBuf_
  const auto tMap0 = std::chrono::steady_clock::now();
  {
    const int offY = (pH - H) / 2;
    const int offX = (pW - W) / 2;
    float *mx = reinterpret_cast<float *>(mapX_.data);
    float *my = reinterpret_cast<float *>(mapY_.data);
    float *defX = defBuf_;
    float *defY = defBuf_ + N2;
    int HW = H * W;

#ifdef USE_MPS
    if (!gRenderPipeline) {
      metal::init();
      gRenderPipeline = metal::createPipeline("renderLensedImage", kLensingShader);
    }
    if (!gRemapPipeline) {
      gRemapPipeline = metal::createPipeline("buildRemapMaps", kLensingShader);
    }
    ensureRenderBuffers(W, H, pW, pH);
    if (gRenderPipeline && gRenderBufs && currentBackground_.isContinuous() &&
        latestLensed_.isContinuous()) {
      struct { int w, h, pw, ox, oy; float s; bool di; } params = {
          W, H, pW, offX, offY, strength_, distortInside_};
      const void *data[] = {defX, defY, mask.data, currentBackground_.data,
                            latestLensed_.data, &params};
      const bool uploadMask[] = {true, true, true, gRenderBackgroundDirty,
                                 false, true};
      const bool downloadMask[] = {false, false, false, false, true, false};
      metal::dispatchWithBuffersSelective(gRenderPipeline, HW, gRenderBufs,
                                          data, uploadMask, downloadMask, 6);
      gRenderBackgroundDirty = false;
      return;
    }
    ensureRemapBuffers(W, H, pW, pH);
    if (gRemapPipeline && gRemapBufs) {
      struct { int w, h, pw, ox, oy; float s; bool di; } params = {
          W, H, pW, offX, offY, strength_, distortInside_};
      const void *data[] = {defX, defY, mask.data, mx, my, &params};
      const bool uploadMask[] = {true, true, true, false, false, true};
      const bool downloadMask[] = {false, false, false, true, true, false};
      metal::dispatchWithBuffersSelective(gRemapPipeline, HW, gRemapBufs,
                                          data, uploadMask, downloadMask, 6);
    } else
#endif
    {
#pragma omp parallel for num_threads(nthreads_)
      for (int idx = 0; idx < HW; ++idx) {
        int y = idx / W, x = idx - y * W;
        if (!distortInside_ && mask.at<uchar>(y, x) > 0) {
          mx[idx] = float(x);
          my[idx] = float(y);
          continue;
        }
        float dx = defX[(y + offY) * pW + (x + offX)] * strength_;
        float dy = defY[(y + offY) * pW + (x + offX)] * strength_;
        float xx = x + dx, yy = y + dy;
        mx[idx] = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
        my[idx] = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
      }
    }
  }
  const auto tMap1 = std::chrono::steady_clock::now();
  perfMapBuild.addSample(
      std::chrono::duration<double, std::milli>(tMap1 - tMap0).count());

  // Apply the remap to the current background, this will use the maps to
  // interpolate the location of the pixels in the lensed image
  const auto tRemap0 = std::chrono::steady_clock::now();
  cv::remap(currentBackground_, latestLensed_, mapX_, mapY_, cv::INTER_LINEAR,
            cv::BORDER_REFLECT);
  const auto tRemap1 = std::chrono::steady_clock::now();
  perfRemap.addSample(
      std::chrono::duration<double, std::milli>(tRemap1 - tRemap0).count());
}

/**
 * @brief Update the geometry of the lensing worker.
 *
 * This will free and reallocate all the FFTW plans and buffers, to
 * account for the new dimensions.
 *
 * @param width  New width of the lens
 * @param height  New height of the lens
 */
void LensingWorker::updateGeometry(int width, int height) {

  // Update the lens geometry
  width_ = width;
  height_ = height;

  // Update the padded dimensions
  padWidth_ = width_ * padFactor_;
  padHeight_ = height_ * padFactor_;

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

  // Rebuild the mask and lensed images
  latestLensed_.create(height_, width_, CV_8UC3);

  // Release and reallocate the mask and map matrices
  mapX_.create(height_, width_, CV_32FC1);
  mapY_.create(height_, width_, CV_32FC1);

}

/**
 * @brief When we get a new background, update the current background.
 *
 * @param background The new background image.
 */
void LensingWorker::onBackgroundChange(const cv::Mat &background) {
  // Update the current background
  currentBackground_ = background;
  outputWidth_ = background.cols;
  outputHeight_ = background.rows;

  cv::resize(currentBackground_, currentBackground_,
             calculationSize(background.size(), lowerRes_),
             0, 0, cv::INTER_AREA);

#ifdef USE_MPS
  gRenderBackgroundDirty = true;
#endif

  // Update the geometry to match the new background, this will also free and
  // reallocate all the FFTW plans and buffers we need for the lensing effect
  updateGeometry(currentBackground_.cols, currentBackground_.rows);
}

/**
 * @brief When we get a new mask, apply the lensing effect.
 *
 * @param mask The new mask image.
 */
void LensingWorker::onMask(const cv::Mat &mask) {
  static thread_local PerfLog perf("lensing", 60);
  static thread_local PerfLog perfApply("lensing-apply", 60);

  // If we don't have a background, theres nothing to do
  if (currentBackground_.empty()) {
    return;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();

    // Apply the lensing effect
    const auto tApply0 = std::chrono::steady_clock::now();
    applyLensing(mask);
    const auto tApply1 = std::chrono::steady_clock::now();

    if (lowerRes_ < 1.0f) {
      cv::resize(latestLensed_, upsampledLensed_,
                 cv::Size(outputWidth_, outputHeight_),
                 0, 0, cv::INTER_LINEAR);
    } else {
      upsampledLensed_ = latestLensed_;
    }

    // Emit the lensed image
    emit lensedReady(upsampledLensed_.clone());

    const auto t1 = std::chrono::steady_clock::now();
    perf.addSample(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    perfApply.addSample(
        std::chrono::duration<double, std::milli>(tApply1 - tApply0).count());

  } catch (const std::exception &e) {
    emit lensingError("Lensing error: " + std::string(e.what()));
    return;
  }
}
