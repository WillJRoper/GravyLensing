/**
 * @file segmentation_worker.hpp
 *
 * This file defines the worker class used to segment a frame to find
 * people using libtorch. A new mask is generated for every new frame.
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

// Qt includes
#include <QDebug>
#include <QObject>

// Torch includes (with slots override to avoid conflicts with Qt)
#if defined(slots)
#pragma push_macro("slots")
#undef slots
#endif
#include <torch/script.h>
#include <torch/torch.h>
#if defined(slots)
#pragma pop_macro("slots")
#endif

// External includes
#include <opencv2/opencv.hpp>

class SegmentationWorker : public QObject {
  Q_OBJECT

public:
  // ================== Member Variable Declarations ==================

  // ================== Member Function Prototypes ==================

  // Constructor
  SegmentationWorker(const std::string &modelPath, int modelSize = 512,
                     int nthreads = 1, float temporalSmooth = 0.6f,
                     float lowerRes = 1.0f);

  // Check loaded model
  bool isModelLoaded() const { return modelLoaded_; }

  // ===================== Qt Slots ==================

public Q_SLOTS:

  // Calculate a new mask when there is a new frame
  void onFrame(const cv::Mat &frame);

  // Update the geometry when the background changes
  void onBackgroundChange(const cv::Mat &background);

  // ===================== Qt Signals ==================

signals:

  // Signal to emit when the mask is ready
  void maskReady(const cv::Mat &mask);

  // Signal to emit when there is an error in the segmentation
  void segmentationError(const std::string &error);

private:
  // ================== Private Member Variable Declarations ==================

  // Path to the segmentation model
  std::string modelPath_;

  // Internal state for segmentation
  torch::jit::script::Module segmentModel_;

  // The training device
  torch::Device device_;

  // pre-allocated Tensor for inference on the CPU and GPU
  torch::Tensor inputCpuTensor_;
  torch::Tensor inputTensor_;

  // Dimensions for the model
  int fastW_, fastH_;
  int width_, height_;

  // The number of threads we have spare (excluding Qt ones)
  int nthreads_;

  // The lower resolution factor for the lensing effect. The resolution at which
  // the lensing effect is calculed will be this much smaller than the
  // background resolution.
  float lowerRes_;

  // The Matrix to hold the mask
  cv::Mat smallFrame_;
  cv::Mat rgbFrame_;
  cv::Mat fastMask_;
  cv::Mat latestMask_;
  cv::Mat prevPersonProb_;
  cv::Mat smoothMask_;

  // Smoothing factor in [0,1], defining weight between new and old mask
  const float temporalSmooth_ = 0.6f;

  // Flag to indicate if we have a previous probability map
  bool havePrevProb_ = false;

  // Which output channel is “person”?
  static constexpr int kPersonClass_ = 15;

  // Drop bolbs in the mask smaller than this
  const int minBlobArea = 50;

  // Did we load successfully?
  bool modelLoaded_ = false;

  // ================== Member Function Prototypes ==================

  // Detect the person mask in the current frame using a segmentation model.
  void detectPersonMask(const cv::Mat &frame);

  // Set up the segmentation model
  void setupSegmentationModel(const std::string &modelPath);

  // Update the geometry when the background changes
  void updateGeometry(int width, int height);
};

/**
 * @brief Pick the device for PyTorch operations.
 *
 * This function checks for MPS and CUDA availability, and returns the
 * appropriate device.
 *
 * @returns torch::Device object representing the selected device.
 */
static torch::Device pickDevice() {
  torch::DeviceType dtype = torch::kCPU;

#ifdef USE_MPS
  if (torch::mps::is_available()) {
    qInfo() << "[SegmentationWorker] Using MPS backend";
    return torch::Device(torch::kMPS);
  }
#endif

#ifdef USE_CUDA
  if (torch::cuda::is_available()) {
    qInfo() << "[SemgmentationWorker] Using CUDA backend";
    return torch::Device(torch::kCUDA);
  }
#endif

  else {
    qInfo() << "[SemgmentationWorker] Using CPU backend";
    return torch::Device(torch::kCPU);
  }
}
