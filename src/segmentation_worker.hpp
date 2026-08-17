/**
 * @file segmentation_worker.hpp
 *
 * This file defines the worker class used to segment a frame to find people.
 * A new mask is generated for every new frame.
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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include <QRect>

// Qt includes
#include <QDebug>
#include <QObject>

// External includes
#include <opencv2/opencv.hpp>

#include "apple_video_frame.hpp"

class SegmentationWorker : public QObject {
  Q_OBJECT

public:
  // ================== Member Variable Declarations ==================

  // ================== Member Function Prototypes ==================

  // Constructor
  SegmentationWorker(int visionSize = 512, float temporalSmooth = 0.6f,
                     float lowerRes = 1.0f,
                     const std::string &qualityMode = "balanced",
                     int personSensitivity = 50);
  ~SegmentationWorker();

  bool isReady() const { return ready_; }

  // Thread-safe frame submission that coalesces stale work.  seq identifies
  // the captured frame and is carried through to maskReady so consumers can
  // align the mask with the exact camera frame it came from.
  void submitAppleFrame(const AppleVideoFrame &frame,
                        const cv::Mat &guidanceFrame, quint64 seq);

  // Whether this worker should process incoming frames.
  bool isEnabled() const { return enabled_.load(); }

  // ===================== Qt Slots ==================

public Q_SLOTS:

  // Enable or disable mask generation while keeping the worker alive.
  void setEnabled(bool enabled);

  // Update the geometry when the background changes
  void onBackgroundChange(const cv::Mat &background);
  void beginShutdown();

  // ===================== Qt Signals ==================

signals:

  // Signal to emit when the mask is ready.  seq matches the camera frame the
  // mask was derived from.
  void maskReady(const cv::Mat &mask, quint64 seq);

  // Signal to emit when there is an error in the segmentation
  void segmentationError(const std::string &error);

private:
  class ApplePersonSegmentationHelper;

  // ================== Private Member Variable Declarations ==================

  // macOS-native Vision backend.
  std::unique_ptr<ApplePersonSegmentationHelper> appleSegmentationHelper_;
  std::string qualityMode_;

  // Dimensions for the model
  int fastW_, fastH_;
  int width_, height_;
  float lowerRes_;

  // The Matrix to hold the mask
  cv::Mat fastMask_;
  cv::Mat latestMask_;
  cv::Mat prevPersonProb_;
  cv::Mat smoothMask_;
  cv::Mat refinedPersonProb_;
  cv::Mat adaptiveAlpha_;
  cv::Mat motionProbDelta_;
  cv::Mat uncertaintyBand_;

  // ROI acceleration state.
  cv::Rect currentVisionROI_;
  int framesSinceVisionFullFrame_ = 0;
  int visionROIStableFrames_ = 0;

  // Smoothing factor in [0,1], defining weight between new and old mask
  const float temporalSmooth_ = 0.6f;

  // Flag to indicate if we have a previous probability map
  bool havePrevProb_ = false;

  // Drop bolbs in the mask smaller than this
  const int minBlobArea = 50;

  // Thresholds for converting the refined soft mask to a binary mask.
  float personOnThreshold_ = 0.50f;
  float personOffThreshold_ = 0.35f;

  // Temporal median filter on the probability map.
  // Maintains a circular buffer of recent probability maps and uses the
  // per-pixel median instead of the raw frame.  This eliminates single-frame
  // detection dropouts for small/distant people (the median ignores outliers)
  // while preserving real transitions within ~2 frames.
  static constexpr int kMedianWindow = 5;
  std::vector<cv::Mat> probHistory_;
  int probHistoryWriteIdx_ = 0;
  int probHistoryCount_ = 0;

  // Adaptive temporal smoothing parameters.
  const float temporalMinAlpha_ = 0.18f;
  const float temporalMaxAlpha_ = 0.82f;

  // ROI acceleration parameters.
  const int visionFullFrameInterval_ = 12;
  const int visionROIPadding_ = 32;
  const int visionMinROIDim_ = 192;
  const float visionMaxROIAreaFraction_ = 0.65f;



  bool ready_ = false;

  // Whether this worker should process frames.
  std::atomic<bool> enabled_{false};
  std::atomic<bool> shuttingDown_{false};

  // ================== Member Function Prototypes ==================

  void setupVision();

  // Update the geometry when the background changes
  void updateGeometry(int width, int height);

  // Vision inference runs on a dedicated thread so the temporal/refinement
  // chain (on the worker's QThread) overlaps with the next Vision request.
  // The inbox is latest-wins; a single inference slot keeps completions in
  // capture order.
  struct VisionCompletion {
    quint64 seq = 0;
    cv::Mat prob;
    cv::Mat guidance; // shallow, read-only
    cv::Rect activeROI;
    bool usedROI = false;
  };
  struct VisionROISuggestion {
    cv::Rect2f normalizedCrop;
    cv::Rect activeROI;
    int requestWidth = 0;
    int requestHeight = 0;
    bool useROI = false;
  };

  void inferenceLoop();
  void drainCompletions();
  void applyPersonResult(VisionCompletion &&completion);

  std::thread inferenceThread_;
  std::mutex inboxMutex_;
  std::condition_variable inboxCv_;
  AppleVideoFrame inboxFrame_;
  cv::Mat inboxGuidance_;
  quint64 inboxSeq_ = 0;
  bool inboxValid_ = false;

  std::mutex completionMutex_;
  std::deque<VisionCompletion> completions_;
  bool completionDrainScheduled_ = false;

  std::mutex roiSuggestionMutex_;
  VisionROISuggestion roiSuggestion_;

  std::atomic<bool> inferenceStop_{false};
};
