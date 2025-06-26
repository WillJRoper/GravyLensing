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
#include <iostream>
#include <opencv2/imgproc.hpp>

// Local includes
#include "segmentation_worker.hpp"

/**
 * @brief Constructor for the SegmentationWorker class.
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
    : nthreads_(nthreads), lowerRes_(lowerRes),
      temporalSmooth_(temporalSmooth) {

  // Create initial subtractor
  backSub_ = cv::createBackgroundSubtractorKNN(1000, 400, false);
}

void SegmentationWorker::detectPersonMask(const cv::Mat &frame) {
  // 1) KNN subtract
  cv::Mat fgMask;
  backSub_->apply(frame, fgMask, /*learningRate=*/0.01);

  // 2) Morphological clean (optional)
  int msize = 10;
  auto kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, {msize, msize});
  cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kern);
  cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kern);

  // 4) EMA blend into prevMaskFloat_
  if (!havePrevMask_) {
    prevMaskFloat_ = fgMask.clone();
    havePrevMask_ = true;
  } else {
    cv::addWeighted(fgMask, temporalSmooth_, prevMaskFloat_,
                    1.0f - temporalSmooth_, 0.0, prevMaskFloat_);
  }

  // 5) Threshold the smoothed float mask back to binary 0/1
  cv::Mat binSmooth;
  cv::threshold(fgMask, binSmooth, 0.5, 1.0, cv::THRESH_BINARY);

  // 6) Upsample to output size
  cv::resize(binSmooth, latestMask_, latestMask_.size(), 0, 0,
             cv::INTER_NEAREST);

  // 7) Convert to 8-bit (0 or 255) for display or downstream use
  latestMask_.convertTo(latestMask_, CV_8UC1, 255.0);
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
