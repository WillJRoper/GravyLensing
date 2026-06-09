/**
 * @file color_mask.cpp
 *
 * Adaptive color tracking worker for GravyLensing.
 */

#include "color_mask.hpp"

#include <cmath>
#include <iostream>

namespace {
struct ClickSelection {
  cv::Point point{-1, -1};
  bool clicked = false;
};

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
  ready_ = initializeColorModel(initialFrame);
}

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
}

void ColorMaskWorker::ensureFrameBuffers(const cv::Size &size) {
  if (frameSize_ == size) {
    return;
  }

  frameSize_ = size;
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
  if (stats.count == 0 || stats.sat < kMinSaturation_) {
    setError("Selected object color is too desaturated to track robustly");
    return false;
  }

  targetHue_ = stats.hue;
  targetSat_ = stats.sat;
  targetVal_ = stats.val;
  confidence_ = 1.0f;

  std::cout << "[ColorMaskWorker] Selected HSV target: H=" << targetHue_
            << " S=" << targetSat_ << " V=" << targetVal_ << "\n";
  return true;
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
  HSVStats stats;
  double sumX = 0.0;
  double sumY = 0.0;
  double sumS = 0.0;
  double sumV = 0.0;

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
      ++stats.count;
    }
  }

  if (stats.count == 0) {
    return stats;
  }

  float angle = std::atan2(sumY, sumX);
  if (angle < 0.0f) {
    angle += 2.0f * static_cast<float>(CV_PI);
  }

  stats.hue = angle * 180.0f / (2.0f * static_cast<float>(CV_PI));
  stats.sat = static_cast<float>(sumS / stats.count);
  stats.val = static_cast<float>(sumV / stats.count);
  return stats;
}

bool ColorMaskWorker::reacquisitionMode() const {
  return lostFrames_ > 0 && lostFrames_ < kReacquireFrames_;
}

bool ColorMaskWorker::buildCandidateMask(const cv::Mat &frame) {
  ensureFrameBuffers(frame.size());
  cv::cvtColor(frame, hsvFrame_, cv::COLOR_BGR2HSV);

  const int extraHue = reacquisitionMode() ? std::min(lostFrames_ / 3, 10) : 0;
  const int hueTolerance = std::min(kBaseHueTolerance_ + extraHue,
                                    kMaxHueTolerance_);

  for (int y = 0; y < hsvFrame_.rows; ++y) {
    const cv::Vec3b *src = hsvFrame_.ptr<cv::Vec3b>(y);
    uchar *dst = candidateMask_.ptr<uchar>(y);
    for (int x = 0; x < hsvFrame_.cols; ++x) {
      const float hueDist = wrappedHueDistance(src[x][0], targetHue_);
      const int sat = src[x][1];
      const int val = src[x][2];
      const bool inRange = hueDist <= hueTolerance &&
                           sat >= std::max(kMinSaturation_,
                                           static_cast<int>(targetSat_) -
                                               kSatTolerance_) &&
                           std::abs(val - targetVal_) <= kValTolerance_;
      dst[x] = inRange ? 255 : 0;
    }
  }

  cv::morphologyEx(candidateMask_, cleanedMask_, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             cv::Size(kOpenKernel_,
                                                      kOpenKernel_)));
  cv::morphologyEx(cleanedMask_, cleanedMask_, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             cv::Size(kCloseKernel_,
                                                      kCloseKernel_)));
  return true;
}

std::vector<ColorMaskWorker::Candidate> ColorMaskWorker::extractCandidates() {
  std::vector<Candidate> candidates;
  const int numLabels = cv::connectedComponentsWithStats(
      cleanedMask_, labelImage_, stats_, centroids_, 8, CV_32S);

  for (int label = 1; label < numLabels; ++label) {
    Candidate candidate;
    candidate.label = label;
    candidate.area = stats_.at<int>(label, cv::CC_STAT_AREA);
    if (candidate.area < kMinBlobArea_) {
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
  float bestScore = -1.0f;
  runnerUpScore = -1.0f;

  for (auto candidate : candidates) {
    const float areaDelta =
        std::fabs(static_cast<float>(candidate.area - prevArea_)) /
        static_cast<float>(std::max(prevArea_, 1));
    const float areaScore =
        haveTrack_ ? 1.0f - std::min(areaDelta, 1.0f)
                   : clamp01(candidate.area / 2500.0f);
    const float distance = haveTrack_
                               ? cv::norm(candidate.centroid - prevCentroid_)
                               : 0.0f;
    const float distanceScore = haveTrack_ ? 1.0f - std::min(distance / (diag * 0.35f), 1.0f)
                                           : 0.6f;
    const float overlapScore = haveTrack_ ? computeIoU(candidate.bbox, prevBox_) : 0.4f;
    const float borderPenalty = candidate.touchesBorder ? 0.25f : 0.0f;

    candidate.score = 0.40f * areaScore + 0.35f * distanceScore +
                      0.25f * overlapScore - borderPenalty;

    if (candidate.score > bestScore) {
      runnerUpScore = bestScore;
      bestScore = candidate.score;
      bestCandidate = candidate;
    } else if (candidate.score > runnerUpScore) {
      runnerUpScore = candidate.score;
    }
  }

  return bestScore >= kMinConfidenceForTrack_;
}

void ColorMaskWorker::buildSelectedMask(int label) {
  cv::compare(labelImage_, label, selectedMaskFrame_, cv::CMP_EQ);
  selectedMaskFrame_.convertTo(selectedMaskFrame_, CV_8U, 255);
}

bool ColorMaskWorker::updateColorModelIfConfident(const Candidate &candidate,
                                                  float runnerUpScore) {
  if (candidate.score < kMinConfidenceForUpdate_ ||
      (runnerUpScore >= 0.0f && candidate.score - runnerUpScore < kRunnerUpMargin_) ||
      candidate.touchesBorder) {
    return false;
  }

  cv::erode(selectedMaskFrame_, erodedMask_,
            cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                      cv::Size(kErodeKernel_, kErodeKernel_)));
  HSVStats observed = computeMaskedHSVStats(hsvFrame_, erodedMask_);
  if (observed.count == 0 || observed.sat < kMinSaturation_) {
    return false;
  }

  const float hueDelta = wrappedHueDistance(observed.hue, targetHue_);
  if (hueDelta > kUpdateHueDistanceLimit_) {
    return false;
  }

  float signedDelta = observed.hue - targetHue_;
  if (signedDelta > 90.0f) {
    signedDelta -= 180.0f;
  } else if (signedDelta < -90.0f) {
    signedDelta += 180.0f;
  }

  targetHue_ += kColorUpdateAlpha_ * signedDelta;
  if (targetHue_ < 0.0f) {
    targetHue_ += 180.0f;
  } else if (targetHue_ >= 180.0f) {
    targetHue_ -= 180.0f;
  }

  targetSat_ += kColorUpdateAlpha_ * (observed.sat - targetSat_);
  targetVal_ += kColorUpdateAlpha_ * (observed.val - targetVal_);
  return true;
}

void ColorMaskWorker::onFrame(const cv::Mat &frame) {
  if (!ready_ || latestMask_.empty() || frame.empty()) {
    return;
  }

  try {
    buildCandidateMask(frame);
    const std::vector<Candidate> candidates = extractCandidates();

    Candidate bestCandidate;
    float runnerUpScore = -1.0f;
    const bool found = selectBestCandidate(candidates, bestCandidate, runnerUpScore);

    if (!found) {
      ++lostFrames_;
      if (lostFrames_ >= kReacquireFrames_) {
        haveTrack_ = false;
      }
      if (!lastGoodMask_.empty() && lostFrames_ <= kHoldLastMaskFrames_) {
        emit maskReady(lastGoodMask_);
      } else {
        latestMask_.setTo(0);
        emit maskReady(latestMask_);
      }
      return;
    }

    buildSelectedMask(bestCandidate.label);
    cv::resize(selectedMaskFrame_, latestMask_, latestMask_.size(), 0, 0,
               cv::INTER_NEAREST);

    confidence_ = bestCandidate.score;
    prevCentroid_ = bestCandidate.centroid;
    prevBox_ = bestCandidate.bbox;
    prevArea_ = bestCandidate.area;
    haveTrack_ = true;
    lostFrames_ = 0;
    lastGoodMask_ = latestMask_.clone();

    updateColorModelIfConfident(bestCandidate, runnerUpScore);
    emit maskReady(latestMask_);

  } catch (const std::exception &e) {
    setError("Color mask error: " + std::string(e.what()));
  }
}

void ColorMaskWorker::onBackgroundChange(const cv::Mat &background) {
  updateGeometry(background.cols * lowerRes_, background.rows * lowerRes_);
}
