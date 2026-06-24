/**
 * @file color_mask.cpp
 *
 * Color tracking worker for GravyLensing.
 */

#include "color_mask.hpp"
#include "perf_log.hpp"

#ifdef USE_MPS
#include "metal_helper.h"

static const char *kHsvShader = R"(
#include <metal_stdlib>
using namespace metal;

struct HsvParams {
    int width;
    int height;
    float targetHue;
    float targetSat;
    float targetVal;
    int hueTol;
    int satTol;
    int valTol;
    int minSaturation;
};

kernel void thresholdHSV(
    device const uchar *hsv   [[buffer(0)]],
    device uchar *mask        [[buffer(1)]],
    constant HsvParams &p     [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= p.width * p.height) return;
    
    int base = idx * 3;
    float h = float(hsv[base + 0]);
    float s = float(hsv[base + 1]);
    float v = float(hsv[base + 2]);
    
    // Circular hue distance (hue wraps at 0/180)
    float hueDist = abs(h - p.targetHue);
    hueDist = min(hueDist, 180.0f - hueDist);
    
    float rawDist = abs(h - p.targetHue);
    bool hueOk = (hueDist <= float(p.hueTol)) &&
                 (rawDist <= float(p.hueTol) ||
                  (p.targetHue > 30.0f && p.targetHue < 150.0f));
    
    int satLower = max(p.minSaturation, int(p.targetSat) - p.satTol);
    bool inRange = hueOk && int(s) >= satLower &&
                   int(abs(v - p.targetVal)) <= p.valTol;
    
    mask[idx] = inRange ? 255 : 0;
}
)";

static metal::Pipeline *gHsvPipeline = nullptr;
#endif

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
struct ClickSelection {
  cv::Point point{-1, -1};
  bool clicked = false;
};

// Minimal OpenCV mouse callback used during startup color selection.
void onMousePick(int event, int x, int y, int, void *userdata) {
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) {
    return;
  }

  auto *selection = static_cast<ClickSelection *>(userdata);
  selection->point = cv::Point(x, y);
  selection->clicked = true;
}

bool selectColorPoint(const cv::Mat &frame, cv::Point &point) {
  ClickSelection selection;
  const std::string windowName = "Select Lens Color";
  cv::Mat preview = frame.clone();
  std::cout << "[ColorMaskWorker] Waiting for color selection in window '"
            << windowName << "'\n";
  cv::putText(preview,
              "Click target color. ESC/c cancel.",
              cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(255, 255, 255), 2);

  cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(windowName, onMousePick, &selection);

  while (true) {
    cv::imshow(windowName, preview);
    int key = cv::waitKey(20);
    if (selection.clicked) {
      point = selection.point;
      // Show confirmation marker so the user can verify the click landed
      // on the intended target before the window closes.
      cv::circle(preview, point, 12, cv::Scalar(0, 0, 255), 3);
      cv::imshow(windowName, preview);
      cv::waitKey(1500);
      cv::destroyWindow(windowName);
      return true;
    }
    if (key == 27 || key == 'c' || key == 'C') {
      cv::destroyWindow(windowName);
      return false;
    }
  }
}
} // namespace

ColorMaskWorker::ColorMaskWorker(const cv::Mat &initialFrame, float lowerRes)
    : lowerRes_(lowerRes) {
  enabled_ = true;
  ready_ = initializeColorModel(initialFrame);
}

ColorMaskWorker::ColorMaskWorker(float lowerRes) : lowerRes_(lowerRes) {}

ColorMaskWorker::ColorMaskWorker(float hue, float sat, float val,
                                  int width, int height, float lowerRes)
    : lowerRes_(lowerRes),
      targetHue_(hue),
      targetSat_(sat),
      targetVal_(val) {
  enabled_ = true;
  adaptiveHueTol_ = kBaseHueTolerance_;
  adaptiveSatTol_ = kSatTolerance_;
  adaptiveValTol_ = kValTolerance_;
  // Defer geometry creation to caller — matches interactive-path behaviour.
  ensureFrameBuffers(cv::Size(width, height));
  ready_ = true;
}

// Keep error reporting in one place so startup and runtime failures behave the
// same way.
void ColorMaskWorker::setError(const std::string &error) {
  lastError_ = error;
  emit maskError(error);
}

void ColorMaskWorker::updateGeometry(int width, int height) {
  width_ = width;
  height_ = height;
  latestMask_.create(height_, width_, CV_8UC1);
  latestMask_.setTo(0);
  lastGoodMask_.release();
  smoothedMask_.release();
}

void ColorMaskWorker::ensureFrameBuffers(const cv::Size &size) {
  if (frameSize_ == size) {
    return;
  }

  frameSize_ = size;
  scaledFrame_.create(size, CV_8UC3);
  hsvFrame_.create(size, CV_8UC3);
  candidateMask_.create(size, CV_8UC1);
  cleanedMask_.create(size, CV_8UC1);
  selectedMaskFrame_.create(size, CV_8UC1);
  erodedMask_.create(size, CV_8UC1);
  prevBox_ = cv::Rect();
  haveTrack_ = false;
  prevArea_ = 0;
  lostFrames_ = 0;
}

bool ColorMaskWorker::initializeColorModel(const cv::Mat &initialFrame) {
  if (initialFrame.empty()) {
    setError("Initial frame for color selection is empty");
    return false;
  }

  cv::Point point;
  if (!selectColorPoint(initialFrame, point)) {
    setError("Color selection cancelled");
    return false;
  }

  ensureFrameBuffers(initialFrame.size());
  cv::cvtColor(initialFrame, hsvFrame_, cv::COLOR_BGR2HSV);

  int x0 = std::max(0, point.x - kPatchRadius_);
  int y0 = std::max(0, point.y - kPatchRadius_);
  int x1 = std::min(initialFrame.cols, point.x + kPatchRadius_ + 1);
  int y1 = std::min(initialFrame.rows, point.y + kPatchRadius_ + 1);
  cv::Rect patchRect(x0, y0, x1 - x0, y1 - y0);
  if (patchRect.width <= 0 || patchRect.height <= 0) {
    setError("Invalid color selection patch");
    return false;
  }

  cv::Mat patchMask(patchRect.size(), CV_8UC1, cv::Scalar(255));
  HSVStats stats = computeMaskedHSVStats(hsvFrame_(patchRect), patchMask);
  if (stats.count == 0) {
    setError("Color selection patch is empty");
    return false;
  }

  targetHue_ = stats.hue;
  // If the patch straddles a specular highlight the mean saturation can dip
  // very low.  Clamp to a sensible floor so the tracker still discriminates
  // against the background.
  targetSat_ = std::max(stats.sat, (float)kMinSaturation_);
  targetVal_ = stats.val;

  // Keep the keyed colour fixed to the selected sample. Adaptive tolerances
  // drift too easily in cluttered scenes, so use the configured static
  // tolerances for the default fixed-key path.
  adaptiveHueTol_ = kBaseHueTolerance_;
  adaptiveSatTol_ = kSatTolerance_;
  adaptiveValTol_ = kValTolerance_;

  std::cout << "[ColorMaskWorker] Clicked pixel: ("
            << point.x << ", " << point.y << ")\n";
  std::cout << "[ColorMaskWorker] Selected HSV target: H=" << targetHue_
            << " S=" << targetSat_ << " V=" << targetVal_ << "\n";
  std::cout << "[ColorMaskWorker] Selection tolerances: H="
            << adaptiveHueTol_ << " S=" << adaptiveSatTol_
            << " V=" << adaptiveValTol_ << "\n";
  return true;
}

ColorMaskWorker::HSVStats
ColorMaskWorker::runInteractiveColorPicker(const cv::Mat &frame) {
  HSVStats stats{};
  cv::Point point;
  if (!selectColorPoint(frame, point)) {
    return stats; // count == 0 signals cancellation
  }

  int x0 = std::max(0, point.x - kPatchRadius_);
  int y0 = std::max(0, point.y - kPatchRadius_);
  int x1 = std::min(frame.cols, point.x + kPatchRadius_ + 1);
  int y1 = std::min(frame.rows, point.y + kPatchRadius_ + 1);
  cv::Rect patchRect(x0, y0, x1 - x0, y1 - y0);
  if (patchRect.width <= 0 || patchRect.height <= 0) {
    return stats;
  }

  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::Mat patchMask(patchRect.size(), CV_8UC1, cv::Scalar(255));
  stats = computeMaskedHSVStats(hsv(patchRect), patchMask);
  stats.sat = std::max(stats.sat, (float)kMinSaturation_);

  std::cout << "[ColorMaskWorker] Clicked pixel: ("
            << point.x << ", " << point.y << ")\n";
  std::cout << "[ColorMaskWorker] Selected HSV target: H=" << stats.hue
            << " S=" << stats.sat << " V=" << stats.val << "\n";
  return stats;
}

float ColorMaskWorker::wrappedHueDistance(float a, float b) {
  float diff = std::fabs(a - b);
  return std::min(diff, 180.0f - diff);
}

float ColorMaskWorker::clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

float ColorMaskWorker::computeIoU(const cv::Rect &a, const cv::Rect &b) {
  if (a.area() <= 0 || b.area() <= 0) {
    return 0.0f;
  }

  int interArea = (a & b).area();
  if (interArea <= 0) {
    return 0.0f;
  }

  int unionArea = a.area() + b.area() - interArea;
  return unionArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0f;
}

ColorMaskWorker::HSVStats
ColorMaskWorker::computeMaskedHSVStats(const cv::Mat &hsv, const cv::Mat &mask) {
  // First pass: compute means (same as before).
  double sumX = 0.0;
  double sumY = 0.0;
  double sumS = 0.0;
  double sumV = 0.0;
  int count = 0;

  for (int y = 0; y < hsv.rows; ++y) {
    const cv::Vec3b *hsvRow = hsv.ptr<cv::Vec3b>(y);
    const uchar *maskRow = mask.ptr<uchar>(y);
    for (int x = 0; x < hsv.cols; ++x) {
      if (maskRow[x] == 0) {
        continue;
      }
      const float hue = static_cast<float>(hsvRow[x][0]);
      const float angle = hue * 2.0f * static_cast<float>(CV_PI) / 180.0f;
      sumX += std::cos(angle);
      sumY += std::sin(angle);
      sumS += hsvRow[x][1];
      sumV += hsvRow[x][2];
      ++count;
    }
  }

  HSVStats stats;
  stats.count = count;
  if (count == 0) {
    return stats;
  }

  float angle = std::atan2(sumY, sumX);
  if (angle < 0.0f) {
    angle += 2.0f * static_cast<float>(CV_PI);
  }

  stats.hue = angle * 180.0f / (2.0f * static_cast<float>(CV_PI));
  stats.sat = static_cast<float>(sumS / count);
  stats.val = static_cast<float>(sumV / count);

  // Second pass: compute mean absolute deviation from the mean as a robust
  // measure of spread. This is used to set initial per-object tolerances so
  // the mask covers the full object, not just the click region.
  double devX = 0.0, devY = 0.0, devS = 0.0, devV = 0.0;
  const float cosH = std::cos(angle);
  const float sinH = std::sin(angle);

  for (int y = 0; y < hsv.rows; ++y) {
    const cv::Vec3b *hsvRow = hsv.ptr<cv::Vec3b>(y);
    const uchar *maskRow = mask.ptr<uchar>(y);
    for (int x = 0; x < hsv.cols; ++x) {
      if (maskRow[x] == 0) {
        continue;
      }
      const float hue = static_cast<float>(hsvRow[x][0]);
      const float hRad = hue * 2.0f * static_cast<float>(CV_PI) / 180.0f;
      devX += std::fabs(std::cos(hRad) - cosH);
      devY += std::fabs(std::sin(hRad) - sinH);
      devS += std::fabs(hsvRow[x][1] - stats.sat);
      devV += std::fabs(hsvRow[x][2] - stats.val);
    }
  }

  // Approximate hue spread from the mean absolute deviation of the unit vector
  // components.
  const float meanDevXY = static_cast<float>((devX + devY) / (2.0 * count));
  stats.hueSpread = meanDevXY * 90.0f / 1.57f;
  stats.satSpread = static_cast<float>(devS / count);
  stats.valSpread = static_cast<float>(devV / count);

  return stats;
}

bool ColorMaskWorker::reacquisitionMode() const {
  return lostFrames_ > 0;
}

bool ColorMaskWorker::buildCandidateMask(const cv::Mat &frame) {
  ensureFrameBuffers(frame.size());
  cv::cvtColor(frame, hsvFrame_, cv::COLOR_BGR2HSV);

  // Use the tolerances computed at initialization from the selected patch.
  // Fall back to static defaults if init hasn't set them yet.
  const int hueTol = adaptiveHueTol_ > 0 ? adaptiveHueTol_ : kBaseHueTolerance_;
  const int satTol = adaptiveSatTol_ > 0 ? adaptiveSatTol_ : kSatTolerance_;
  const int valTol = adaptiveValTol_ > 0 ? adaptiveValTol_ : kValTolerance_;

  // Loosen the hue tolerance slightly while we are trying to reacquire the
  // object, but keep the range capped so the tracker does not drift too far.
  const int extraHue = reacquisitionMode() ? std::min(lostFrames_ / 3, 10) : 0;
  const int hueTolerance = std::min(hueTol + extraHue, kMaxHueTolerance_);
  const int satLower = std::max(kMinSaturation_,
                                static_cast<int>(targetSat_) - satTol);

  int W = hsvFrame_.cols;
  int H = hsvFrame_.rows;
  int N = W * H;

#ifdef USE_MPS
  if (!gHsvPipeline) {
    metal::init();
    gHsvPipeline = metal::createPipeline("thresholdHSV", kHsvShader);
  }
  if (gHsvPipeline) {
    struct { int w, h; float th, ts, tv; int ht, st, vt, ms; } params = {
        W, H, targetHue_, targetSat_, targetVal_, hueTolerance, satTol,
        valTol, kMinSaturation_};
    const void *bufs[] = {hsvFrame_.data, candidateMask_.data, &params};
    const unsigned long lens[] = {
        (unsigned long)(N * 3 * sizeof(uchar)),
        (unsigned long)(N * sizeof(uchar)),
        sizeof(params)};
    metal::dispatch(gHsvPipeline, N, bufs, lens, 3);
  } else
#endif
  {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
      const cv::Vec3b *src = hsvFrame_.ptr<cv::Vec3b>(y);
      uchar *dst = candidateMask_.ptr<uchar>(y);
      const int cols = W;
      for (int x = 0; x < cols; ++x) {
        const float hueDist = wrappedHueDistance(src[x][0], targetHue_);
        const int sat = src[x][1];
        const int val = src[x][2];
        const float rawDist =
            std::fabs(static_cast<float>(src[x][0]) - targetHue_);
        const bool hueOk =
            (hueDist <= hueTolerance) &&
            (rawDist <= hueTolerance ||
             (targetHue_ > 30.0f && targetHue_ < 150.0f));
        const bool inRange = hueOk && sat >= satLower &&
                              std::abs(val - targetVal_) <= valTol;
        dst[x] = inRange ? 255 : 0;
      }
    }
  }

  static const cv::Mat kOpenKernelMat =
      cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                cv::Size(kOpenKernel_, kOpenKernel_));
  static const cv::Mat kCloseKernelMat =
      cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                cv::Size(kCloseKernel_, kCloseKernel_));
  cv::morphologyEx(candidateMask_, cleanedMask_, cv::MORPH_OPEN,
                   kOpenKernelMat);
  cv::morphologyEx(cleanedMask_, cleanedMask_, cv::MORPH_CLOSE,
                   kCloseKernelMat);
  return true;
}

std::vector<ColorMaskWorker::Candidate> ColorMaskWorker::extractCandidates() {
  std::vector<Candidate> candidates;
  const int numLabels = cv::connectedComponentsWithStats(
      cleanedMask_, labelImage_, stats_, centroids_, 8, CV_32S);
  const int minBlobArea = std::max(
      50, static_cast<int>(std::round(kMinBlobArea_ * lowerRes_ * lowerRes_)));

  for (int label = 1; label < numLabels; ++label) {
    Candidate candidate;
    candidate.label = label;
    candidate.area = stats_.at<int>(label, cv::CC_STAT_AREA);
    if (candidate.area < minBlobArea) {
      continue;
    }

    candidate.bbox = cv::Rect(stats_.at<int>(label, cv::CC_STAT_LEFT),
                              stats_.at<int>(label, cv::CC_STAT_TOP),
                              stats_.at<int>(label, cv::CC_STAT_WIDTH),
                              stats_.at<int>(label, cv::CC_STAT_HEIGHT));
    candidate.centroid = cv::Point2f(
        static_cast<float>(centroids_.at<double>(label, 0)),
        static_cast<float>(centroids_.at<double>(label, 1)));
    candidate.touchesBorder = candidate.bbox.x <= 0 || candidate.bbox.y <= 0 ||
                              candidate.bbox.br().x >= cleanedMask_.cols - 1 ||
                              candidate.bbox.br().y >= cleanedMask_.rows - 1;
    candidates.push_back(candidate);
  }

  return candidates;
}

bool ColorMaskWorker::selectBestCandidate(const std::vector<Candidate> &candidates,
                                          Candidate &bestCandidate,
                                          float &runnerUpScore) {
  if (candidates.empty()) {
    return false;
  }

  const float diag =
      std::sqrt(static_cast<float>(cleanedMask_.cols * cleanedMask_.cols +
                                   cleanedMask_.rows * cleanedMask_.rows));
  const float frameArea =
      static_cast<float>(cleanedMask_.cols * cleanedMask_.rows);
  float bestScore = -1.0f;
  runnerUpScore = -1.0f;

  for (auto candidate : candidates) {
    const float areaDelta =
        std::fabs(static_cast<float>(candidate.area - prevArea_)) /
        static_cast<float>(std::max(prevArea_, 1));
    const float areaScore =
        haveTrack_ ? 1.0f - std::min(areaDelta, 1.0f)
                   : clamp01(candidate.area / (frameArea * 0.01f));
    const float distance = haveTrack_
                               ? cv::norm(candidate.centroid - prevCentroid_)
                               : 0.0f;
    const float distanceScore =
        haveTrack_ ? 1.0f - std::min(distance / (diag * 0.55f), 1.0f) : 0.6f;
    const float overlapScore =
        haveTrack_ ? computeIoU(candidate.bbox, prevBox_) : 0.0f;

    // No border penalty during acquisition — objects often re-enter
    // from the frame edge.
    const float borderPenalty =
        (haveTrack_ && candidate.touchesBorder) ? 0.25f : 0.0f;

    if (haveTrack_) {
      candidate.score = 0.40f * areaScore + 0.35f * distanceScore +
                        0.25f * overlapScore - borderPenalty;
    } else {
      // Acquisition: prefer any reasonably-sized blob.  No distance
      // or overlap heuristics apply since we have no track history.
      candidate.score = 1.0f * areaScore - borderPenalty;
    }

    if (candidate.score > bestScore) {
      runnerUpScore = bestScore;
      bestScore = candidate.score;
      bestCandidate = candidate;
    } else if (candidate.score > runnerUpScore) {
      runnerUpScore = candidate.score;
    }
  }

  // During acquisition, reject candidates whose centroid colour doesn't
  // actually match the target — prevents latching onto wrong objects.
  if (!haveTrack_) {
    const int cx =
        std::clamp(static_cast<int>(bestCandidate.centroid.x), 0,
                    hsvFrame_.cols - 1);
    const int cy =
        std::clamp(static_cast<int>(bestCandidate.centroid.y), 0,
                    hsvFrame_.rows - 1);
    const cv::Vec3b &pix = hsvFrame_.at<cv::Vec3b>(cy, cx);
    const float hueDist = wrappedHueDistance(pix[0], targetHue_);
    if (hueDist > kBaseHueTolerance_ * 2)
      return false;
  }

  return bestScore >= (haveTrack_ ? kMinConfidenceForTrack_ : 0.10f);
}

void ColorMaskWorker::buildSelectedMask(int label) {
  cv::compare(labelImage_, label, selectedMaskFrame_, cv::CMP_EQ);
  selectedMaskFrame_.convertTo(selectedMaskFrame_, CV_8U, 255);
}

void ColorMaskWorker::triggerReselect() {
  if (enabled_) {
    requestReselect_ = true;
    std::cout << "[ColorMaskWorker] Reselect requested\n";
  }
}

void ColorMaskWorker::setTolerances(int hue, int sat, int val) {
  adaptiveHueTol_ = std::max(1, hue);
  adaptiveSatTol_ = std::max(1, sat);
  adaptiveValTol_ = std::max(1, val);
}

void ColorMaskWorker::applyReselectionTarget(float hue, float sat, float val,
                                             int hueTol, int satTol, int valTol,
                                             bool success) {
  if (!success) {
    if (!ready_) {
      latestMask_.setTo(0);
      lastGoodMask_.release();
      smoothedMask_.release();
      emit selectionStateChanged(false);
    }
    return;
  }

  targetHue_ = hue;
  targetSat_ = sat;
  targetVal_ = val;
  (void)hueTol;
  (void)satTol;
  (void)valTol;
  adaptiveHueTol_ = kBaseHueTolerance_;
  adaptiveSatTol_ = kSatTolerance_;
  adaptiveValTol_ = kValTolerance_;
  ready_ = true;
  haveTrack_ = false;
  lostFrames_ = 0;
  lastGoodMask_.release();
  smoothedMask_.release();

  std::cout << "[ColorMaskWorker] Reselected target: H=" << targetHue_
            << " S=" << targetSat_ << " V=" << targetVal_ << "\n";
  std::cout << "[ColorMaskWorker] Selection tolerances: H="
            << adaptiveHueTol_ << " S=" << adaptiveSatTol_ << " V="
            << adaptiveValTol_ << "\n";
  emit selectionStateChanged(true);
}

void ColorMaskWorker::setEnabled(bool enabled) {
  if (!enabled_ && enabled) {
    frameCount_ = 0;
  }
  enabled_ = enabled;
}

void ColorMaskWorker::onFrame(const cv::Mat &frame) {
  static thread_local PerfLog perf("color-mask", 60);
  if (!enabled_ || latestMask_.empty() || frame.empty()) {
    return;
  }

  // Re-pick colour on user request. Emit a cross-thread signal so that the
  // picker window opens on the main thread (required by macOS AppKit).
  if (requestReselect_) {
    requestReselect_ = false;
    emit reselectionRequested(frame.clone());
    return;
  }

  if (!ready_) {
    return;
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    const cv::Size procSize = latestMask_.size();
    if (procSize.width > 0 && procSize.height > 0 && frame.size() != procSize) {
      cv::resize(frame, scaledFrame_, procSize, 0, 0, cv::INTER_LINEAR);
      buildCandidateMask(scaledFrame_);
    } else {
      buildCandidateMask(frame);
    }

    if (!trackedBlobMode_) {
      cv::resize(cleanedMask_, latestMask_, latestMask_.size(), 0, 0,
                 cv::INTER_NEAREST);

      // Light open to kill single-pixel noise.
      cv::morphologyEx(latestMask_, latestMask_, cv::MORPH_OPEN,
                       cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                 cv::Size(3, 3)));

      // Area-based blob filter — precise, and fast when the mask is
      // sparse (uses boundingRect area as proxy instead of contourArea).
      const int minArea = std::max(
          20, static_cast<int>(std::round(kMinBlobArea_ * lowerRes_ *
                                          lowerRes_)));
      if (cv::countNonZero(latestMask_) > minArea) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(latestMask_, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        latestMask_.setTo(0);
        for (const auto &c : contours) {
          if (cv::boundingRect(c).area() >= minArea)
            cv::drawContours(latestMask_,
                             std::vector<std::vector<cv::Point>>{c}, 0,
                             cv::Scalar(255), cv::FILLED);
        }
      }

      if (smoothedMask_.empty()) {
        smoothedMask_ = latestMask_.clone();
      } else {
        cv::addWeighted(latestMask_, kMaskSmoothAlpha_, smoothedMask_,
                        1.0f - kMaskSmoothAlpha_, 0, smoothedMask_);
      }

      emit maskReady(smoothedMask_.clone());

      const auto t1 = std::chrono::steady_clock::now();
      perf.addSample(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
      ++frameCount_;
      return;
    }

    const std::vector<Candidate> candidates = extractCandidates();

    Candidate bestCandidate;
    float runnerUpScore = -1.0f;
    const bool found = selectBestCandidate(candidates, bestCandidate, runnerUpScore);

    const bool acquired = found && !haveTrack_;
    const bool lost = !found && haveTrack_;
    const bool justLost = lost && lostFrames_ == 0;
    const bool periodic = (frameCount_ % 30) == 0;

    if (frameCount_ < 5 || periodic || acquired || justLost) {
      int rawPx = cv::countNonZero(candidateMask_);
      int cleanedPx = cv::countNonZero(cleanedMask_);
      std::cout << "[ColorMaskWorker] frame " << frameCount_
                << "  rawPx=" << rawPx
                << "  cleanPx=" << cleanedPx
                << "  cands=" << candidates.size()
                << "  targetH=" << targetHue_
                << " S=" << targetSat_ << " V=" << targetVal_;
      if (found) {
        std::cout << "  bestScore=" << bestCandidate.score
                  << "  bestArea=" << bestCandidate.area
                  << "  runnerUp=" << runnerUpScore;
      } else {
        std::cout << "  NO-CANDIDATE  lost=" << lostFrames_;
      }
      if (justLost) {
        std::cout << "  ** LOST **";
      } else if (acquired) {
        std::cout << "  ** ACQUIRED **";
      }
      std::cout << "\n";
    }

    if (!found) {
      ++lostFrames_;
      if (lostFrames_ >= kReacquireFrames_) {
        haveTrack_ = false;
      }
      // Hold the last good mask briefly to hide one-frame dropouts.
      if (!lastGoodMask_.empty() && lostFrames_ <= kHoldLastMaskFrames_) {
        lastGoodMask_.copyTo(latestMask_);
      } else {
        latestMask_.setTo(0);
      }
    } else {
      buildSelectedMask(bestCandidate.label);
      cv::resize(selectedMaskFrame_, latestMask_, latestMask_.size(), 0, 0,
                 cv::INTER_NEAREST);

      prevCentroid_ = bestCandidate.centroid;
      prevBox_ = bestCandidate.bbox;
      prevArea_ = bestCandidate.area;
      haveTrack_ = true;
      lostFrames_ = 0;
      lastGoodMask_ = latestMask_.clone();
    }

    // Exponential moving average over the emitted mask to suppress
    // per-frame jitter and sudden blob switches.
    if (smoothedMask_.empty()) {
      smoothedMask_ = latestMask_.clone();
    } else {
      cv::addWeighted(latestMask_, kMaskSmoothAlpha_,
                      smoothedMask_, 1.0f - kMaskSmoothAlpha_, 0,
                      smoothedMask_);
    }
    emit maskReady(smoothedMask_.clone());

    const auto t1 = std::chrono::steady_clock::now();
    perf.addSample(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

  } catch (const std::exception &e) {
    setError("Color mask error: " + std::string(e.what()));
  }
  ++frameCount_;
}

void ColorMaskWorker::onBackgroundChange(const cv::Mat &background) {
  updateGeometry(background.cols * lowerRes_, background.rows * lowerRes_);
}
