/**
 * LensMask
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
#pragma once

#include "cam_feed.hpp"

#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

/**
 * LensMask
 *
 * This class defines a mask derived from a webcam image. This mask contains
 * all pixels that are part of a person in the frame (derived from using
 * torch detection).
 *
 * Once derived the masked pixel define a "gravitational lens" that can be
 * used to apply a lensing effect to a background image. The lensing effect
 * is computed using FFTs of the deflection kernels, which are precomputed
 * and cached for efficiency.
 *
 */
class LensMask {
public:
  // Threading for async segmentation
  std::thread workerThread_;
  std::atomic<bool> stopWorker_{false};

  // Pointer to the CameraFeed object so we can access the latest frame
  CameraFeed *camFeed_ = nullptr;

  // Create a flag for whether there is a new mask to show
  std::atomic<bool> newMaskReady_{false};

  // Create a flag for whether there is a new lensed image to show
  std::atomic<bool> newLensedImageReady_{false};

  // The latest mask computed
  cv::Mat latestMask_;
  std::mutex maskMutex_; // guards latestMask_

  // The latest lensed image computed
  cv::Mat latestLensed_;
  std::mutex lensedMutex_; // guards latestLensed_

  /**
   * @brief Constructor
   *
   * @param camFeed   Camera feed object to get frames from
   * @param softening  Core softening radius to avoid singular deflections
   * @param padFactor  Padding factor for FFT to reduce edge artifacts
   * @param strength   Mass scaling for deflection magnitude
   */
  LensMask(CameraFeed *camFeed, float softening = 30.0f, int padFactor = 2,
           float strength = 1.0f, const std::string &modelPath = "",
           int maskScale = 4);

  /** Destructor: cleans up FFT plans and buffers */
  ~LensMask();

  // Worker entrypoint
  void segmentationLoop();

  /**
   * @brief Detect the person mask in the current frame using a segmentation
   * model.
   *
   * @param frame  Input BGR image
   */
  void detectPersonMask();

  void shadeMask(cv::Mat &frame);

  /**
   * @brief Precompute and cache the FFTs of the deflection kernels.
   *
   * Must be called after construction and before applyLensing().
   */
  void buildKernels();

  /**
   * @brief Apply gravitational lensing to a static background using a binary
   * mask.
   *
   * @param background   CV_8UC3 BGR background image
   * @param output       CV_8UC3 BGR output image (same size as background)
   */
  void applyLensing(const cv::Mat &background);

  /// Start the segmentation worker thread
  void startAsyncSegmentation();

  /// Stop the worker and join
  void stopAsyncSegmentation();

private:
  int width_, height_;       // Frame dimensions
  int padWidth_, padHeight_; // FFT padded dimensions
  int padFactor_;            // How much to pad (e.g. 2×)
  float softening_;          // Softening radius
  float strength_;           // Mass scaling
  int maskScale_;            // how much we downsample for the segmentation
  int fastW_, fastH_;        // dimensions for the fast segmentation image

  // Internal state for segmentation
  torch::jit::script::Module segmentModel_;
  torch::Device device_;

  // pre-allocated Tensor for inference
  torch::Tensor inputTensor_;

  // OpenCV VideoCapture for webcam
  cv::VideoCapture cap_;

  // FFTW buffers and plans for kernel transforms
  float *kernelX_ = nullptr;
  float *kernelY_ = nullptr;
  fftwf_complex *Kx_ft_ = nullptr;
  fftwf_complex *Ky_ft_ = nullptr;
  fftwf_plan planKx_;
  fftwf_plan planKy_;

  // FFT buffers/plans for mask and deflection
  float *maskBuf_ = nullptr;        // size padH*padW
  fftwf_complex *maskFT_ = nullptr; // size padH*(padW/2+1)
  fftwf_complex *defXFT_ = nullptr; // same size
  fftwf_complex *defYFT_ = nullptr; // same size
  float *defXBuf_ = nullptr;        // size padH*padW
  float *defYBuf_ = nullptr;        // size padH*padW

  fftwf_plan planMask_; // r2c on maskBuf_
  fftwf_plan planDefX_; // c2r for defXBuf_
  fftwf_plan planDefY_; // c2r for defYBuf_
};
