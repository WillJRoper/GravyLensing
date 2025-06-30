/**
 * @file segmentation_worker.cpp
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

// Standard includes
#include <chrono>
#include <iostream>

// Local includes
#include "segmentation_worker.hpp"

/**
 * @brief Constructor for the SegmentationWorker class.
 *
 * This constructor initializes the segmentation model and sets up the device
 * for PyTorch operations.
 *
 * @param modelPath The path to the segmentation model.
 * @param modelSize The size of the model (default is 512).
 * @param nthreads The number of threads to use (default is 1).
 * @param temporalSmooth The smoothing factor for temporal frames [0,1]
 * @param lowerRes The lower resolution factor for the lensing effect. The
 *   resolution at which the lensing effect is calculed will be this much
 *   smaller than the background resolution.
 */
SegmentationWorker::SegmentationWorker(const std::string &modelPath,
                                       int modelSize, int nthreads,
                                       float temporalSmooth, float lowerRes)
    : modelPath_(modelPath), fastW_(modelSize), fastH_(modelSize),
      nthreads_(nthreads), device_(pickDevice()),
      temporalSmooth_(temporalSmooth), lowerRes_(lowerRes) {

  std::cout << "[SegmentationWorker] Initializing segmentation model...\n";
  std::cout << "[SegmentationWorker] Using device: " << device_ << "\n";
  std::cout << "[SegmentationWorker] Model size: " << fastW_ << "x" << fastH_
            << "\n";
  std::cout << "[SegmentationWorker] Temporal smoothing: " << temporalSmooth_
            << "\n";

  // We need to set up the fixed size tensors we'll need for the model
  smallFrame_.create(fastH_, fastW_, CV_8UC3);
  rgbFrame_.create(fastH_, fastW_, CV_8UC3);
  fastMask_.create(fastH_, fastW_, CV_8UC1);
  prevPersonProb_.create(fastH_, fastW_, CV_8UC1);
  smoothMask_.create(fastH_, fastW_, CV_8UC1);

  // Set up the segmentation model
  setupSegmentationModel(modelPath);

  // Exit if the loading failed
  if (!modelLoaded_) {
    emit segmentationError("Failed to load the segmentation model from " +
                           modelPath);
    return;
  }

  std::cout << "[SegmentationWorker] Loaded model from " << modelPath_ << "\n";
}

/**
 * @brief Set up the segmentation model.
 *
 * This function loads the segmentation model from the specified path and
 * prepares it for inference.
 *
 * @param modelPath The path to the segmentation model.
 */
void SegmentationWorker::setupSegmentationModel(const std::string &modelPath) {
  try {
    // Load the segmentation model
    segmentModel_ = torch::jit::load(modelPath_, device_);
    segmentModel_.to(device_);
    segmentModel_.eval();
    modelLoaded_ = true;
  } catch (const c10::Error &e) {
    modelLoaded_ = false;
    return;
  }

  // Allocate the device tensor (empty for now)
  inputTensor_ = torch::empty(
      {1, 3, fastH_, fastW_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

#ifdef USE_MPS
  // Allocate a CPU tensor for staging
  inputCpuTensor_ = torch::empty(
      {1, 3, fastH_, fastW_},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
#endif
}

void SegmentationWorker::detectPersonMask(const cv::Mat &frame) {

  // Downsample the frame to the model size
  cv::resize(frame, smallFrame_, cv::Size(fastW_, fastH_), 0, 0,
             cv::INTER_LINEAR);

  // Convert from BGR to RGB for Torch
  cv::cvtColor(smallFrame_, rgbFrame_, cv::COLOR_BGR2RGB);

#ifdef USE_MPS

  // COPY raw pixels into CPU staging tensor (no CPU normalization)
  float *cpu_ptr = inputCpuTensor_.data_ptr<float>();
  const int HW = fastH_ * fastW_;
#pragma omp parallel for num_threads(nthreads_)
  for (int y = 0; y < fastH_; ++y) {
    const cv::Vec3b *row = rgbFrame_.ptr<cv::Vec3b>(y);
    for (int x = 0; x < fastW_; ++x) {
      int idx = y * fastW_ + x;
      cpu_ptr[0 * HW + idx] = row[x][0] / 255.f;
      cpu_ptr[1 * HW + idx] = row[x][1] / 255.f;
      cpu_ptr[2 * HW + idx] = row[x][2] / 255.f;
    }
  }

  // COPY CPU staging → GPU device tensor
  inputTensor_.copy_(inputCpuTensor_, /*non_blocking=*/true);

  // NORMALIZE in-place on MPS
  {
    torch::NoGradGuard no_grad;
    inputTensor_[0][0].sub_(0.485f).div_(0.229f);
    inputTensor_[0][1].sub_(0.456f).div_(0.224f);
    inputTensor_[0][2].sub_(0.406f).div_(0.225f);
  }

  // RUN inference on GPU
  static const auto forwardMethod = segmentModel_.get_method("forward");
  auto out_iv = forwardMethod({inputTensor_});

  // Unwrap IValue → logits tensor:
  torch::Tensor logits;
  if (out_iv.isTensor())
    logits = out_iv.toTensor();
  else if (out_iv.isTuple())
    logits = out_iv.toTuple()->elements()[0].toTensor();
  else if (out_iv.isGenericDict())
    logits = out_iv.toGenericDict().at("out").toTensor();
  else {
    std::cerr << "[SegmentationWorker] Bad IValue\n";
    return;
  }

  // Bring logits back to CPU, pick class
  torch::Tensor probs = logits.squeeze(0).softmax(0);
  torch::Tensor personProb_t = probs[kPersonClass_].to(torch::kCPU);

#else

  // Copy into pre-allocated tensor and normalize to [0,1]
  float *tptr = inputTensor_.data_ptr<float>();
#pragma omp parallel for num_threads(nthreads_)
  for (int y = 0; y < fastH_; ++y) {
    const cv::Vec3b *row = rgbFrame_.ptr<cv::Vec3b>(y);
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

  // Run the model
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

  // Convert logits → class map
  torch::Tensor probs = logits.squeeze(0).softmax(0);
  torch::Tensor personProb_t = probs[dPersonClass_];

#endif

  // Convert to an OpenCV Mat (CV_32F)
  cv::Mat newPersonProb(fastH_, fastW_, CV_32F,
                        (void *)personProb_t.data_ptr<float>());

  if (!havePrevProb_) {
    prevPersonProb_ = newPersonProb.clone();
    havePrevProb_ = true;
  } else {
    cv::addWeighted(newPersonProb, temporalSmooth_, prevPersonProb_,
                    1.0f - temporalSmooth_, 0.0, prevPersonProb_);
  }

  cv::threshold(prevPersonProb_, smoothMask_, 0.5, 255, cv::THRESH_BINARY);
  smoothMask_.convertTo(fastMask_, CV_8U);

  cv::morphologyEx(fastMask_, fastMask_, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(fastMask_, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  for (auto &c : contours) {
    if (cv::contourArea(c) < minBlobArea)
      cv::drawContours(fastMask_, std::vector<std::vector<cv::Point>>{c}, 0, 0,
                       -1);
  }

  cv::resize(fastMask_, latestMask_, latestMask_.size(), 0, 0,
             cv::INTER_NEAREST);
}

/**
 * @brief Update the geometry of the segmentation worker.
 *
 * This function updates the geometry of the segmentation worker when the
 * background changes.
 *
 * Note that the model geometry is fixed, so we don't need to do anything
 * there.
 *
 * @param width The new width of the background.
 * @param height The new height of the background.
 */
void SegmentationWorker::updateGeometry(int width, int height) {
  // Update the dimensions
  width_ = width;
  height_ = height;

  // (Re)Allocate the buffers for the segmentation model
  latestMask_.create(height_, width_, CV_8UC1);
}

/**
 * @brief When we get a new frame, update the segmentation model with it.
 *
 * This function is called when a new frame is received from the camera feed.
 *
 * @param frame The new frame from the camera feed.
 */
void SegmentationWorker::onFrame(const cv::Mat &frame) {

  // Nothing to do until a background has been set
  if (latestMask_.empty()) {
    return;
  }

  try {

    // Detect the person mask in the current frame
    detectPersonMask(frame);

    // Emit the mask ready signal
    emit maskReady(latestMask_);

  } catch (const std::exception &e) {
    emit segmentationError("Segmentation error: " + std::string(e.what()));
    return;
  }
}

/**
 * @brief When the background changes, update the segmentation model.
 *
 * @param background The new background image.
 */
void SegmentationWorker::onBackgroundChange(const cv::Mat &background) {
  // Update the geometry to match the new background
  updateGeometry(background.cols * lowerRes_, background.rows * lowerRes_);
}
