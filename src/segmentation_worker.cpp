/**
 * @file segmentation_worker.cpp
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

// Standard includes
#include <algorithm>
#include <chrono>
#include <iostream>

// Local includes
#include "vision_segmentation_helper.hpp"
#include "perf_log.hpp"
#include "processing_geometry.hpp"
#include "segmentation_worker.hpp"

namespace {

double elapsedMs(const std::chrono::steady_clock::time_point &start,
                 const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void refineSoftPersonMask(const cv::Mat &personProb, const cv::Mat &guidanceFrame,
                          cv::Mat &refinedProb, cv::Mat &binaryMask,
                          float onThreshold, float offThreshold,
                          int minBlobArea) {
  static const cv::Mat kGrowKernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
  static constexpr int kWeakGrowIterations = 4;
  static constexpr float kEdgeBarrierThreshold = 0.35f;

  cv::GaussianBlur(personProb, refinedProb, cv::Size(5, 5), 0.0, 0.0,
                   cv::BORDER_REPLICATE);

  cv::Mat strongForeground;
  cv::Mat weakForeground;
  cv::threshold(refinedProb, strongForeground, onThreshold, 255,
                cv::THRESH_BINARY);
  cv::threshold(refinedProb, weakForeground, offThreshold, 255,
                cv::THRESH_BINARY);
  strongForeground.convertTo(strongForeground, CV_8U);
  weakForeground.convertTo(weakForeground, CV_8U);

  if (!guidanceFrame.empty()) {
    cv::Mat guidanceSmall;
    if (guidanceFrame.cols != refinedProb.cols ||
        guidanceFrame.rows != refinedProb.rows) {
      cv::resize(guidanceFrame, guidanceSmall,
                 cv::Size(refinedProb.cols, refinedProb.rows), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      guidanceSmall = guidanceFrame;
    }

    cv::Mat guidanceGray;
    if (guidanceSmall.channels() == 3) {
      cv::cvtColor(guidanceSmall, guidanceGray, cv::COLOR_BGR2GRAY);
    } else if (guidanceSmall.channels() == 4) {
      cv::cvtColor(guidanceSmall, guidanceGray, cv::COLOR_BGRA2GRAY);
    } else {
      guidanceGray = guidanceSmall;
    }

    cv::GaussianBlur(guidanceGray, guidanceGray, cv::Size(3, 3), 0.0, 0.0,
                     cv::BORDER_REPLICATE);

    cv::Mat gradX;
    cv::Mat gradY;
    cv::Mat gradMag;
    cv::Sobel(guidanceGray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(guidanceGray, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);

    double gradMax = 0.0;
    cv::minMaxLoc(gradMag, nullptr, &gradMax);
    if (gradMax > 1e-6) {
      gradMag.convertTo(gradMag, CV_32F, 1.0 / gradMax);

      cv::Mat uncertainBand;
      cv::subtract(weakForeground, strongForeground, uncertainBand);

      cv::Mat strongHalo;
      cv::dilate(strongForeground, strongHalo, kGrowKernel);

      cv::Mat edgeBarrierFloat;
      cv::threshold(gradMag, edgeBarrierFloat, kEdgeBarrierThreshold, 255.0,
                    cv::THRESH_BINARY);
      cv::Mat edgeBarrier;
      edgeBarrierFloat.convertTo(edgeBarrier, CV_8U);
      cv::bitwise_and(edgeBarrier, uncertainBand, edgeBarrier);
      cv::bitwise_not(strongHalo, strongHalo);
      cv::bitwise_and(edgeBarrier, strongHalo, edgeBarrier);
      weakForeground.setTo(0, edgeBarrier);
    }
  }

  binaryMask = strongForeground.clone();
  for (int i = 0; i < kWeakGrowIterations; ++i) {
    cv::dilate(binaryMask, binaryMask, kGrowKernel);
    cv::bitwise_and(binaryMask, weakForeground, binaryMask);
  }

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  const int componentCount = cv::connectedComponentsWithStats(
      binaryMask, labels, stats, centroids, 8, CV_32S);
  binaryMask.setTo(cv::Scalar(0));

  for (int label = 1; label < componentCount; ++label) {
    const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int y = stats.at<int>(label, cv::CC_STAT_TOP);
    const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    const bool touchesBorder =
        x == 0 || y == 0 || (x + width) >= binaryMask.cols ||
        (y + height) >= binaryMask.rows;
    const int effectiveMinBlobArea =
        touchesBorder ? std::max(1, minBlobArea / 3) : minBlobArea;
    if (stats.at<int>(label, cv::CC_STAT_AREA) < effectiveMinBlobArea) {
      continue;
    }
    cv::Mat componentMask = labels == label;
    cv::bitwise_or(binaryMask, componentMask, binaryMask);
  }

  cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
}

void adaptiveTemporalBlend(const cv::Mat &newPersonProb, cv::Mat &historyProb,
                           cv::Mat &adaptiveAlpha, cv::Mat &motionProbDelta,
                           cv::Mat &uncertaintyBand, float baseAlpha,
                           float minAlpha, float maxAlpha, float onThreshold,
                           float offThreshold) {
  static const cv::Mat kBandKernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});

  cv::absdiff(newPersonProb, historyProb, motionProbDelta);
  motionProbDelta.convertTo(adaptiveAlpha, CV_32F, 0.5);

  cv::inRange(newPersonProb, cv::Scalar(offThreshold), cv::Scalar(onThreshold),
              uncertaintyBand);
  cv::dilate(uncertaintyBand, uncertaintyBand, kBandKernel);

  cv::Mat uncertaintyFloat;
  uncertaintyBand.convertTo(uncertaintyFloat, CV_32F, 1.0 / 255.0);

  adaptiveAlpha += uncertaintyFloat * 0.10f;
  adaptiveAlpha += baseAlpha;
  cv::min(adaptiveAlpha, maxAlpha, adaptiveAlpha);
  cv::max(adaptiveAlpha, minAlpha, adaptiveAlpha);

  cv::Mat blendedNew;
  cv::Mat blendedHistory;
  cv::Mat invAlpha;
  cv::subtract(cv::Scalar(1.0f), adaptiveAlpha, invAlpha);
  cv::multiply(newPersonProb, adaptiveAlpha, blendedNew);
  cv::multiply(historyProb, invAlpha, blendedHistory);
  cv::add(blendedNew, blendedHistory, historyProb);
}

cv::Rect paddedMaskBounds(const cv::Mat &mask, int padding, cv::Size limit) {
  std::vector<cv::Point> nonZeroPoints;
  cv::findNonZero(mask, nonZeroPoints);
  if (nonZeroPoints.empty()) {
    return cv::Rect();
  }

  cv::Rect bounds = cv::boundingRect(nonZeroPoints);
  bounds.x = std::max(0, bounds.x - padding);
  bounds.y = std::max(0, bounds.y - padding);
  bounds.width = std::min(limit.width - bounds.x, bounds.width + 2 * padding);
  bounds.height =
      std::min(limit.height - bounds.y, bounds.height + 2 * padding);
  return bounds;
}

cv::Rect paddedProbabilityBounds(const cv::Mat &probability, float threshold,
                                 int padding, cv::Size limit) {
  if (probability.empty()) {
    return cv::Rect();
  }

  cv::Mat activeMask;
  cv::threshold(probability, activeMask, threshold, 255.0, cv::THRESH_BINARY);
  activeMask.convertTo(activeMask, CV_8U);
  return paddedMaskBounds(activeMask, padding, limit);
}

float rectIoU(const cv::Rect &a, const cv::Rect &b) {
  const cv::Rect intersection = a & b;
  if (intersection.empty()) {
    return 0.0f;
  }
  const float intersectionArea = static_cast<float>(intersection.area());
  const float unionArea = static_cast<float>(a.area() + b.area() - intersection.area());
  return unionArea > 0.0f ? intersectionArea / unionArea : 0.0f;
}

float meanProbabilityInRect(const cv::Mat &probability, const cv::Rect &roi) {
  if (probability.empty() || roi.empty()) {
    return 0.0f;
  }
  return static_cast<float>(cv::mean(probability(roi))[0]);
}

/// Per-pixel median of N float probability maps (N = 5).
/// Simple insertion sort — fast for such a tiny fixed input.
static void computePixelMedian(const std::vector<cv::Mat> &history,
                               cv::Mat &out) {
  constexpr int N = 5;
  out.create(history[0].size(), CV_32F);
  const int rows = out.rows;
  const int cols = out.cols;

  for (int y = 0; y < rows; ++y) {
    const float *srcRows[N];
    for (int i = 0; i < N; ++i)
      srcRows[i] = history[i].ptr<float>(y);
    float *outRow = out.ptr<float>(y);

    for (int x = 0; x < cols; ++x) {
      float vals[N];
      for (int i = 0; i < N; ++i)
        vals[i] = srcRows[i][x];
      for (int i = 1; i < N; ++i) {
        const float key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
          vals[j + 1] = vals[j];
          --j;
        }
        vals[j + 1] = key;
      }
      outRow[x] = vals[N / 2];
    }
  }
}

} // namespace

/**
 * @brief Constructor for the SegmentationWorker class.
 *
 * This constructor initializes Apple Vision person segmentation.
 *
 * @param visionSize The Vision request size (default is 512).
 * @param temporalSmooth The smoothing factor for temporal frames [0,1]
 */
SegmentationWorker::SegmentationWorker(int visionSize,
                                       float temporalSmooth, float lowerRes,
                                       const std::string &qualityMode,
                                       int personSensitivity)
    : qualityMode_(qualityMode), fastW_(visionSize), fastH_(visionSize),
      lowerRes_(lowerRes),
      temporalSmooth_(temporalSmooth) {

  const float sensitivity =
      static_cast<float>(std::clamp(personSensitivity, 0, 100)) / 100.0f;
  personOnThreshold_ = 0.65f - 0.30f * sensitivity;
  personOffThreshold_ = personOnThreshold_ - 0.15f;

  std::cout << "[SegmentationWorker] Initializing segmentation backend...\n";
  std::cout << "[SegmentationWorker] Model size: " << fastW_ << "x" << fastH_
            << "\n";
  std::cout << "[SegmentationWorker] Temporal smoothing: " << temporalSmooth_
            << "\n";

  // We need to set up the fixed size tensors we'll need for the model
  fastMask_.create(fastH_, fastW_, CV_8UC1);
  prevPersonProb_.create(fastH_, fastW_, CV_32F);
  smoothMask_.create(fastH_, fastW_, CV_8UC1);
  refinedPersonProb_.create(fastH_, fastW_, CV_32F);
  adaptiveAlpha_.create(fastH_, fastW_, CV_32F);
  motionProbDelta_.create(fastH_, fastW_, CV_32F);
  uncertaintyBand_.create(fastH_, fastW_, CV_8UC1);

  // Pre-allocate median-filter circular buffer.
  probHistory_.resize(kMedianWindow);
  for (auto &m : probHistory_)
    m.create(fastH_, fastW_, CV_32F);

  setupVision();

  if (!ready_) {
    emit segmentationError("Apple Vision person segmentation is unavailable");
    return;
  }

  std::cout << "[SegmentationWorker] Segmentation backend ready\n";
}

SegmentationWorker::~SegmentationWorker() = default;

void SegmentationWorker::submitAppleFrame(const AppleVideoFrame &frame) {
  submitAppleFrame(frame, cv::Mat());
}

void SegmentationWorker::submitAppleFrame(const AppleVideoFrame &frame,
                                          const cv::Mat &guidanceFrame) {
  if (!frame.isValid() || shuttingDown_) {
    return;
  }

  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    pendingAppleFrame_ = frame;
    pendingAppleGuidanceFrame_ = guidanceFrame.clone();
    if (!pendingAppleFrameDrainScheduled_) {
      pendingAppleFrameDrainScheduled_ = true;
      shouldSchedule = true;
    }
  }

  if (shouldSchedule) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingAppleFrame(); },
                              Qt::QueuedConnection);
  }
}
void SegmentationWorker::drainPendingAppleFrame() {
  if (shuttingDown_) {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    pendingAppleFrame_ = AppleVideoFrame();
    pendingAppleGuidanceFrame_.release();
    pendingAppleFrameDrainScheduled_ = false;
    return;
  }

  AppleVideoFrame frame;
  cv::Mat guidanceFrame;
  {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    if (!pendingAppleFrame_.isValid()) {
      pendingAppleFrameDrainScheduled_ = false;
      return;
    }
    frame = pendingAppleFrame_;
    guidanceFrame = std::move(pendingAppleGuidanceFrame_);
    pendingAppleFrame_ = AppleVideoFrame();
    pendingAppleGuidanceFrame_.release();
  }

  static thread_local PerfLog perf("person-mask", 60);

  if (!enabled_ || !ready_ || latestMask_.empty()) {
    // If we consumed a frame but are not ready to process it we must
    // clear the drain-scheduled flag; otherwise submitAppleFrame will
    // never schedule the next drain and the pipeline deadlocks.
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    pendingAppleFrameDrainScheduled_ = false;
    return;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    if (detectPersonMask(frame, guidanceFrame)) {
      emit maskReady(latestMask_.clone());
    }
    const auto t1 = std::chrono::steady_clock::now();
    perf.addSample(elapsedMs(t0, t1));
  } catch (const std::exception &e) {
    emit segmentationError("Segmentation error: " + std::string(e.what()));
    return;
  }

  bool shouldContinue = false;
  {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    if (!pendingAppleFrame_.isValid()) {
      pendingAppleFrameDrainScheduled_ = false;
    } else {
      shouldContinue = true;
    }
  }

  if (shouldContinue) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingAppleFrame(); },
                              Qt::QueuedConnection);
  }
}

void SegmentationWorker::setupVision() {
  appleSegmentationHelper_ =
      std::make_unique<ApplePersonSegmentationHelper>();
  if (appleSegmentationHelper_ != nullptr &&
      appleSegmentationHelper_->isAvailable()) {
    appleSegmentationHelper_->setQualityMode(qualityMode_);
    ready_ = true;
    qInfo() << "[SegmentationWorker] Using Vision person segmentation backend";
    return;
  }
  appleSegmentationHelper_.reset();
  ready_ = false;
}

bool SegmentationWorker::detectPersonMask(const AppleVideoFrame &frame,
                                          const cv::Mat &guidanceFrame) {
  static thread_local PerfLog visionPerf("person-mask-vision", 60);
  static thread_local PerfLog cleanupPerf("person-mask-cleanup", 60);
  static thread_local PerfLog upscalePerf("person-mask-upscale", 60);

  auto stageStart = std::chrono::steady_clock::now();
  cv::Mat roiPersonProb;
  cv::Mat newPersonProb;
  if (havePrevProb_) {
    newPersonProb = prevPersonProb_.clone();
  } else {
    newPersonProb = cv::Mat::zeros(fastH_, fastW_, CV_32F);
  }
  std::string error;
  const bool enableVisionROIAcceleration =
      qualityMode_ == "fast" || qualityMode_ == "balanced";
  const cv::Mat &roiSourceProb = havePrevProb_ ? prevPersonProb_ : refinedPersonProb_;

  const cv::Size visionSize(fastW_, fastH_);
  const cv::Rect suggestedROI = paddedProbabilityBounds(
      roiSourceProb, 0.18f, visionROIPadding_, visionSize);
  const float suggestedROIMeanProb = meanProbabilityInRect(roiSourceProb, suggestedROI);
  const float suggestedROIIoU = currentVisionROI_.empty()
                                    ? 1.0f
                                    : rectIoU(suggestedROI, currentVisionROI_);
  const bool shouldUseROI =
      enableVisionROIAcceleration &&
      !suggestedROI.empty() &&
      static_cast<float>(suggestedROI.area()) /
              static_cast<float>(visionSize.area()) <
          visionMaxROIAreaFraction_ &&
      suggestedROIMeanProb >= 0.12f &&
      (currentVisionROI_.empty() || suggestedROIIoU >= 0.35f ||
       visionROIStableFrames_ < 2) &&
      framesSinceVisionFullFrame_ < visionFullFrameInterval_;

  cv::Rect activeROI(0, 0, fastW_, fastH_);
  int requestWidth = fastW_;
  int requestHeight = fastH_;
  cv::Rect2f normalizedCrop;
  if (shouldUseROI) {
    activeROI = currentVisionROI_.empty() ? suggestedROI
                                          : (currentVisionROI_ | suggestedROI);
    activeROI &= cv::Rect(0, 0, fastW_, fastH_);
    requestWidth = std::max(visionMinROIDim_, activeROI.width);
    requestHeight = std::max(visionMinROIDim_, activeROI.height);
    requestWidth = std::min(requestWidth, fastW_);
    requestHeight = std::min(requestHeight, fastH_);
    normalizedCrop = cv::Rect2f(
        static_cast<float>(activeROI.x) / static_cast<float>(fastW_),
        static_cast<float>(activeROI.y) / static_cast<float>(fastH_),
        static_cast<float>(activeROI.width) / static_cast<float>(fastW_),
        static_cast<float>(activeROI.height) / static_cast<float>(fastH_));
  }

  const bool ok = shouldUseROI
                      ? appleSegmentationHelper_->generatePersonProbability(
                            frame, normalizedCrop, requestWidth, requestHeight,
                            roiPersonProb, error)
                      : appleSegmentationHelper_->generatePersonProbability(
                            frame, fastW_, fastH_, roiPersonProb, error);
  if (!ok) {
    std::cerr << "[SegmentationWorker] Vision segmentation failed: " << error
              << "\n";
    return false;
  }

  if (shouldUseROI) {
    cv::Mat resizedROIProb;
    if (roiPersonProb.cols != activeROI.width ||
        roiPersonProb.rows != activeROI.height) {
      cv::resize(roiPersonProb, resizedROIProb, activeROI.size(), 0, 0,
                 cv::INTER_LINEAR);
    } else {
      resizedROIProb = roiPersonProb;
    }
    resizedROIProb.copyTo(newPersonProb(activeROI));
    currentVisionROI_ = activeROI;
    ++framesSinceVisionFullFrame_;
  } else {
    newPersonProb = roiPersonProb;
    currentVisionROI_ = cv::Rect(0, 0, fastW_, fastH_);
    framesSinceVisionFullFrame_ = 0;
  }

  auto stageEnd = std::chrono::steady_clock::now();
  visionPerf.addSample(elapsedMs(stageStart, stageEnd));

  stageStart = stageEnd;
  probHistory_[probHistoryWriteIdx_++] = newPersonProb.clone();
  if (probHistoryWriteIdx_ == kMedianWindow) probHistoryWriteIdx_ = 0;
  if (probHistoryCount_ < kMedianWindow) ++probHistoryCount_;

  cv::Mat blendProb;
  if (probHistoryCount_ >= kMedianWindow) {
    computePixelMedian(probHistory_, blendProb);
  } else {
    blendProb = newPersonProb;
  }
  if (!havePrevProb_) {
    blendProb.copyTo(prevPersonProb_);
    havePrevProb_ = true;
  } else {
    adaptiveTemporalBlend(blendProb, prevPersonProb_, adaptiveAlpha_,
                          motionProbDelta_, uncertaintyBand_,
                          1.0f - temporalSmooth_, temporalMinAlpha_,
                          temporalMaxAlpha_, personOnThreshold_,
                          personOffThreshold_);
  }

  const cv::Mat &activeGuidanceFrame = qualityMode_ == "fast"
                                           ? cv::Mat()
                                           : guidanceFrame;
  refineSoftPersonMask(prevPersonProb_, activeGuidanceFrame, refinedPersonProb_,
                       fastMask_,
                       personOnThreshold_, personOffThreshold_, minBlobArea);
  stageEnd = std::chrono::steady_clock::now();
  cleanupPerf.addSample(elapsedMs(stageStart, stageEnd));

  if (shouldUseROI) {
    const cv::Rect refinedROI = paddedProbabilityBounds(
        prevPersonProb_, 0.15f, visionROIPadding_, visionSize);
    const float refinedMeanProb = meanProbabilityInRect(prevPersonProb_, activeROI);
    if (refinedROI.empty() || refinedMeanProb < 0.08f) {
      currentVisionROI_ = cv::Rect(0, 0, fastW_, fastH_);
      framesSinceVisionFullFrame_ = visionFullFrameInterval_;
      visionROIStableFrames_ = 0;
    } else {
      currentVisionROI_ = activeROI | refinedROI;
      currentVisionROI_ &= cv::Rect(0, 0, fastW_, fastH_);
      ++visionROIStableFrames_;
    }
  } else {
    visionROIStableFrames_ = 0;
  }

  stageStart = stageEnd;
  cv::resize(fastMask_, latestMask_, latestMask_.size(), 0, 0,
             cv::INTER_NEAREST);
  stageEnd = std::chrono::steady_clock::now();
  upscalePerf.addSample(elapsedMs(stageStart, stageEnd));
  return true;
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

  // (Re)allocate the output mask.
  latestMask_.create(height_, width_, CV_8UC1);
}

void SegmentationWorker::setEnabled(bool enabled) { enabled_ = enabled; }

void SegmentationWorker::beginShutdown() {
  shuttingDown_ = true;
  enabled_ = false;
  {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    pendingAppleFrame_ = AppleVideoFrame();
    pendingAppleGuidanceFrame_.release();
    pendingAppleFrameDrainScheduled_ = false;
  }
}

/**
 * @brief When the background changes, update segmentation geometry.
 *
 * @param background The new background image.
 */
void SegmentationWorker::onBackgroundChange(const cv::Mat &background) {
  // Update the geometry to match the new background
  const cv::Size size = calculationSize(background.size(), lowerRes_);
  updateGeometry(size.width, size.height);
}
