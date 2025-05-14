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

// Local includes
#include "lensing_worker.hpp"

/**
 * @brief Constructor for the LensingWorker class.
 *
 * @param strength The strength of the lensing effect.
 * @param softening The softening parameter for the lensing effect.
 * @param padFactor The padding factor for the lensing effect.
 * @param nthreads The number of threads to use for processing.
 * @param lowerRes The lower resolution factor for the lensing effect. The
 *   resolution at which the lensing effect is calculed will be this much
 *   smaller than the background resolution.
 */
LensingWorker::LensingWorker(float strength, float softening, int padFactor,
                             int nthreads, float lowerRes)
    : strength_(strength), softening_(softening), padFactor_(padFactor),
      nthreads_(nthreads), lowerRes_(lowerRes) {

  std::cout << "[LensingWorker] Initializing lensing worker...\n";
  std::cout << "[LensingWorker] Strength: " << strength_ << "\n";
  std::cout << "[LensingWorker] Softening: " << softening_ << "\n";
  std::cout << "[LensingWorker] Padding factor: " << padFactor_ << "\n";
  std::cout
      << "[LensingWorker] Number of threads (excluding those taken by Qt): "
      << nthreads_ << "\n";
  std::cout << "[LensingWorker] Lower resolution factor: " << lowerRes_ << "\n";
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
  // Get some helpful constants shorthands
  const int H = height_;
  const int W = width_;
  const int pH = padHeight_;
  const int pW = padWidth_;
  const int pWC = pW / 2 + 1;
  const int N1 = pH * pWC; // complex bins per deflection
  const int N2 = pH * pW;  // real samples per deflection
  const float inv255 = 1.f / 255.f;

  // Fill maskBuf_ with reflect padding (OpenCV version)
  {
    // Compute borders
    const int top = (pH - H) / 2;
    const int bottom = pH - H - top;
    const int left = (pW - W) / 2;
    const int right = pW - W - left;

    // Pad into an 8-bit mat
    cv::copyMakeBorder(mask, padded_, top, bottom, left, right,
                       cv::BORDER_REFLECT_101);

    // Convert to float and normalize
    padded_.convertTo(paddedF_, CV_32F, inv255);
  }

  // Forward FFT of mask
  fftwf_execute(planMask_);

  // Multiply in Fourier space into defFT_ blocks
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

  // Batched inverse FFT → defBuf_ contains [X; Y]
  fftwf_execute(planDef_);

  // Build remap maps from defBuf_
  {
    const int offY = (pH - H) / 2;
    const int offX = (pW - W) / 2;
    float *mx = reinterpret_cast<float *>(mapX_.data);
    float *my = reinterpret_cast<float *>(mapY_.data);
    float *defX = defBuf_;      // first N2 floats
    float *defY = defBuf_ + N2; // next N2 floats

    int HW = H * W;
#pragma omp parallel for num_threads(nthreads_)
    for (int idx = 0; idx < HW; ++idx) {
      int y = idx / W, x = idx - y * W;
      float dx = defX[(y + offY) * pW + (x + offX)] * strength_;
      float dy = defY[(y + offY) * pW + (x + offX)] * strength_;
      float xx = x + dx, yy = y + dy;
      // clamp
      mx[idx] = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
      my[idx] = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
    }
  }

  // Apply the remap to the current background, this will use the maps to
  // interpolate the location of the pixels in the lensed image
  cv::remap(currentBackground_, latestLensed_, mapX_, mapY_, cv::INTER_LINEAR,
            cv::BORDER_REFLECT);
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

  // Reallocate the padded matrices
  padded_.create(padHeight_, padWidth_, CV_32FC1);
  paddedF_ = cv::Mat(padHeight_, padWidth_, CV_32F, maskBuf_);
}

/**
 * @brief When we get a new background, update the current background.
 *
 * @param background The new background image.
 */
void LensingWorker::onBackgroundChange(const cv::Mat &background) {
  // Update the current background
  currentBackground_ = background;

  // Shrink the background
  cv::resize(currentBackground_, currentBackground_,
             cv::Size(currentBackground_.cols * lowerRes_,
                      currentBackground_.rows * lowerRes_),
             0, 0, cv::INTER_LINEAR);

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

  // If we don't have a background, theres nothing to do
  if (currentBackground_.empty()) {
    return;
  }

  try {

    // Apply the lensing effect
    applyLensing(mask);

    // Resample back to the original size
    cv::Mat upsampledLensed_;
    cv::resize(latestLensed_, upsampledLensed_,
               cv::Size(latestLensed_.cols / lowerRes_,
                        latestLensed_.rows / lowerRes_),
               0, 0, cv::INTER_LINEAR);

    // Emit the lensed image
    emit lensedReady(upsampledLensed_);

  } catch (const std::exception &e) {
    emit lensingError("Lensing error: " + std::string(e.what()));
    return;
  }
}
