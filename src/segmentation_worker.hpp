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

  // The number of threads we have spare (excluding Qt ones)
  int nthreads_;

  // The lower resolution factor for the lensing effect. The resolution at which
  // the lensing effect is calculed will be this much smaller than the
  // background resolution.
  float lowerRes_;

  // The Matrix to hold the mask
  cv::Mat latestMask_;

  // Flag to indicate if we have a previous probability map
  bool havePrevProb_ = false;

  // Which output channel is “person”?
  static constexpr int kPersonClass_ = 15;

  // Drop bolbs in the mask smaller than this
  const int minBlobArea = 50;

  // OpenCV background subtractor
  cv::Ptr<cv::BackgroundSubtractorKNN> backSub_;

  // width and height of the current background
  int width_ = 0;
  int height_ = 0;

  // Temporal smoothing state
  cv::Mat prevMaskFloat_; // CV_32F, same size as latestMask_, values in [0,1]
  bool havePrevMask_ = false;  // whether prevMaskFloat_ is valid
  const float temporalSmooth_; // blend factor α

  // ================== Member Function Prototypes ==================

  // Detect the person mask in the current frame using a segmentation model.
  void detectPersonMask(const cv::Mat &frame);

  // Update the geometry when the background changes
  void updateGeometry(int width, int height);

  // Re-create the subtractor (called onBackgroundChange)
  void resetSubtractor();
};
