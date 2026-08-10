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
#ifdef __APPLE__
#include "vision_segmentation_helper.hpp"
#endif
#include "perf_log.hpp"
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
                                       float temporalSmooth, float lowerRes,
                                       const std::string &qualityMode)
    : modelPath_(modelPath), fastW_(modelSize), fastH_(modelSize),
      nthreads_(nthreads), device_(pickDevice()),
      lowerRes_(lowerRes), qualityMode_(qualityMode),
      temporalSmooth_(temporalSmooth) {

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

SegmentationWorker::~SegmentationWorker() = default;

void SegmentationWorker::submitFrame(const cv::Mat &frame) {
  if (frame.empty() || shuttingDown_) {
    return;
  }

  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lock(pendingFrameMutex_);
    pendingFrame_ = frame;
    if (!pendingFrameDrainScheduled_) {
      pendingFrameDrainScheduled_ = true;
      shouldSchedule = true;
    }
  }

  if (shouldSchedule) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingFrame(); },
                              Qt::QueuedConnection);
  }
}

void SegmentationWorker::submitGuidanceFrame(const cv::Mat &frame) {
  if (frame.empty() || shuttingDown_) {
    return;
  }

  std::lock_guard<std::mutex> lock(guidanceFrameMutex_);
  latestGuidanceFrame_ = frame.clone();
}

#ifdef __APPLE__
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
#endif

void SegmentationWorker::drainPendingFrame() {
  if (shuttingDown_) {
    std::lock_guard<std::mutex> lock(pendingFrameMutex_);
    pendingFrame_.release();
    pendingFrameDrainScheduled_ = false;
    return;
  }

  cv::Mat frame;
  {
    std::lock_guard<std::mutex> lock(pendingFrameMutex_);
    if (pendingFrame_.empty()) {
      pendingFrameDrainScheduled_ = false;
      return;
    }
    frame = std::move(pendingFrame_);
    pendingFrame_.release();
  }

  onFrame(frame);

  bool shouldContinue = false;
  {
    std::lock_guard<std::mutex> lock(pendingFrameMutex_);
    if (pendingFrame_.empty()) {
      pendingFrameDrainScheduled_ = false;
    } else {
      shouldContinue = true;
    }
  }

  if (shouldContinue) {
    QMetaObject::invokeMethod(this, [this]() { drainPendingFrame(); },
                              Qt::QueuedConnection);
  }
}

#ifdef __APPLE__
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

  if (!enabled_ || !modelLoaded_ || latestMask_.empty()) {
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
#endif

/**
 * @brief Set up the segmentation model.
 *
 * This function loads the segmentation model from the specified path and
 * prepares it for inference.
 *
 * @param modelPath The path to the segmentation model.
 */
void SegmentationWorker::setupSegmentationModel(const std::string &modelPath) {
#ifdef __APPLE__
  appleSegmentationHelper_ =
      std::make_unique<ApplePersonSegmentationHelper>();
  if (appleSegmentationHelper_ != nullptr &&
      appleSegmentationHelper_->isAvailable()) {
    appleSegmentationHelper_->setQualityMode(qualityMode_);
    usingAppleVision_ = true;
    modelLoaded_ = true;
    qInfo() << "[SegmentationWorker] Using Vision person segmentation backend";
    return;
  }
  appleSegmentationHelper_.reset();
#endif

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

bool SegmentationWorker::detectPersonMask(const cv::Mat &frame) {
  static thread_local PerfLog visionPerf("person-mask-vision", 60);
  static thread_local PerfLog resizePerf("person-mask-resize", 60);
  static thread_local PerfLog colorPerf("person-mask-color", 60);
  static thread_local PerfLog packPerf("person-mask-pack", 60);
  static thread_local PerfLog uploadPerf("person-mask-upload", 60);
  static thread_local PerfLog normalizePerf("person-mask-normalize", 60);
  static thread_local PerfLog inferPerf("person-mask-infer", 60);
  static thread_local PerfLog probPerf("person-mask-prob", 60);
  static thread_local PerfLog cleanupPerf("person-mask-cleanup", 60);
  static thread_local PerfLog upscalePerf("person-mask-upscale", 60);

  auto stageStart = std::chrono::steady_clock::now();

#ifdef __APPLE__
  if (usingAppleVision_ && appleSegmentationHelper_ != nullptr) {
    return false;
  }
#endif

  // Downsample the frame to the model size
  cv::resize(frame, smallFrame_, cv::Size(fastW_, fastH_), 0, 0,
             cv::INTER_LINEAR);
  auto stageEnd = std::chrono::steady_clock::now();
  resizePerf.addSample(elapsedMs(stageStart, stageEnd));

  // Convert from BGR to RGB for Torch
  stageStart = stageEnd;
  cv::cvtColor(smallFrame_, rgbFrame_, cv::COLOR_BGR2RGB);
  stageEnd = std::chrono::steady_clock::now();
  colorPerf.addSample(elapsedMs(stageStart, stageEnd));

#ifdef USE_MPS

  // COPY raw pixels into CPU staging tensor (no CPU normalization)
  stageStart = stageEnd;
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
  stageEnd = std::chrono::steady_clock::now();
  packPerf.addSample(elapsedMs(stageStart, stageEnd));

  // COPY CPU staging → GPU device tensor
  stageStart = stageEnd;
  inputTensor_.copy_(inputCpuTensor_, /*non_blocking=*/true);
  stageEnd = std::chrono::steady_clock::now();
  uploadPerf.addSample(elapsedMs(stageStart, stageEnd));

  // NORMALIZE in-place on MPS
  stageStart = stageEnd;
  {
    torch::NoGradGuard no_grad;
    inputTensor_[0][0].sub_(0.485f).div_(0.229f);
    inputTensor_[0][1].sub_(0.456f).div_(0.224f);
    inputTensor_[0][2].sub_(0.406f).div_(0.225f);
  }
  stageEnd = std::chrono::steady_clock::now();
  normalizePerf.addSample(elapsedMs(stageStart, stageEnd));

  // RUN inference on GPU
  stageStart = stageEnd;
  static const auto forwardMethod = segmentModel_.get_method("forward");
  auto out_iv = forwardMethod({inputTensor_});
  stageEnd = std::chrono::steady_clock::now();
  inferPerf.addSample(elapsedMs(stageStart, stageEnd));

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
    return false;
  }

  // Bring logits back to CPU, pick class
  stageStart = stageEnd;
  torch::Tensor probs = logits.squeeze(0).softmax(0);
  torch::Tensor personProb_t = probs[kPersonClass_].to(torch::kCPU);
  stageEnd = std::chrono::steady_clock::now();
  probPerf.addSample(elapsedMs(stageStart, stageEnd));

#else

  // Copy into pre-allocated tensor and normalize to [0,1]
  stageStart = stageEnd;
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
  stageEnd = std::chrono::steady_clock::now();
  packPerf.addSample(elapsedMs(stageStart, stageEnd));

  stageStart = stageEnd;
  auto tt = inputTensor_;
  tt[0][0].sub_(0.485f).div_(0.229f);
  tt[0][1].sub_(0.456f).div_(0.224f);
  tt[0][2].sub_(0.406f).div_(0.225f);
  stageEnd = std::chrono::steady_clock::now();
  normalizePerf.addSample(elapsedMs(stageStart, stageEnd));

  // Run the model
  stageStart = stageEnd;
  torch::NoGradGuard no_grad;
  auto out_iv = segmentModel_.forward({inputTensor_});
  stageEnd = std::chrono::steady_clock::now();
  inferPerf.addSample(elapsedMs(stageStart, stageEnd));

  torch::Tensor logits;
  if (out_iv.isTensor())
    logits = out_iv.toTensor();
  else if (out_iv.isTuple())
    logits = out_iv.toTuple()->elements()[0].toTensor();
  else if (out_iv.isGenericDict())
    logits = out_iv.toGenericDict().at("out").toTensor();
  else {
    std::cerr << "Unexpected IValue from segmentation\n";
    return false;
  }

  // Convert logits → class map
  stageStart = stageEnd;
  torch::Tensor probs = logits.squeeze(0).softmax(0);
  torch::Tensor personProb_t = probs[kPersonClass_];
  stageEnd = std::chrono::steady_clock::now();
  probPerf.addSample(elapsedMs(stageStart, stageEnd));

#endif

  // Convert to an OpenCV Mat (CV_32F)
  cv::Mat newPersonProb(fastH_, fastW_, CV_32F,
                        (void *)personProb_t.data_ptr<float>());

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

  refineSoftPersonMask(prevPersonProb_, frame, refinedPersonProb_, fastMask_,
                       personOnThreshold_, personOffThreshold_, minBlobArea);
  stageEnd = std::chrono::steady_clock::now();
  cleanupPerf.addSample(elapsedMs(stageStart, stageEnd));

  stageStart = stageEnd;
  cv::resize(fastMask_, latestMask_, latestMask_.size(), 0, 0,
             cv::INTER_NEAREST);
  stageEnd = std::chrono::steady_clock::now();
  upscalePerf.addSample(elapsedMs(stageStart, stageEnd));
  return true;
}

#ifdef __APPLE__
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

  const cv::Size modelSize(fastW_, fastH_);
  const cv::Rect suggestedROI = paddedProbabilityBounds(
      roiSourceProb, 0.18f, visionROIPadding_, modelSize);
  const float suggestedROIMeanProb = meanProbabilityInRect(roiSourceProb, suggestedROI);
  const float suggestedROIIoU = currentVisionROI_.empty()
                                    ? 1.0f
                                    : rectIoU(suggestedROI, currentVisionROI_);
  const bool shouldUseROI =
      enableVisionROIAcceleration &&
      !suggestedROI.empty() &&
      static_cast<float>(suggestedROI.area()) /
              static_cast<float>(modelSize.area()) <
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
        prevPersonProb_, 0.15f, visionROIPadding_, modelSize);
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
#endif

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
  static thread_local PerfLog perf("person-mask", 60);

  // Nothing to do until a background has been set or this mode is active.
  if (!enabled_ || !modelLoaded_ || latestMask_.empty()) {
    return;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();

    // Detect the person mask in the current frame
    if (detectPersonMask(frame)) {
      // Emit the mask ready signal
      emit maskReady(latestMask_.clone());
    }

    const auto t1 = std::chrono::steady_clock::now();
    perf.addSample(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

  } catch (const std::exception &e) {
    emit segmentationError("Segmentation error: " + std::string(e.what()));
    return;
  }
}

void SegmentationWorker::setEnabled(bool enabled) { enabled_ = enabled; }

void SegmentationWorker::beginShutdown() {
  shuttingDown_ = true;
  enabled_ = false;
  {
    std::lock_guard<std::mutex> lock(pendingFrameMutex_);
    pendingFrame_.release();
    pendingFrameDrainScheduled_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(guidanceFrameMutex_);
    latestGuidanceFrame_.release();
  }
#ifdef __APPLE__
  {
    std::lock_guard<std::mutex> lock(pendingAppleFrameMutex_);
    pendingAppleFrame_ = AppleVideoFrame();
    pendingAppleGuidanceFrame_.release();
    pendingAppleFrameDrainScheduled_ = false;
  }
#endif
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
