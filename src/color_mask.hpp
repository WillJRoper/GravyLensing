/**
 * @file color_mask.hpp
 *
 * Adaptive color tracking worker for GravyLensing.
 */

#pragma once

#include <string>
#include <vector>

#include <QObject>

#include <opencv2/opencv.hpp>

class ColorMaskWorker : public QObject {
  Q_OBJECT

public:
  ColorMaskWorker(const cv::Mat &initialFrame, float lowerRes = 1.0f);

  bool isReady() const { return ready_; }
  const std::string &lastError() const { return lastError_; }

public Q_SLOTS:
  void onFrame(const cv::Mat &frame);
  void onBackgroundChange(const cv::Mat &background);

signals:
  void maskReady(const cv::Mat &mask);
  void maskError(const std::string &error);

private:
  struct Candidate {
    int label = 0;
    int area = 0;
    cv::Rect bbox;
    cv::Point2f centroid;
    bool touchesBorder = false;
    float score = 0.0f;
  };

  struct HSVStats {
    float hue = 0.0f;
    float sat = 0.0f;
    float val = 0.0f;
    int count = 0;
  };

  void updateGeometry(int width, int height);
  void ensureFrameBuffers(const cv::Size &size);
  bool initializeColorModel(const cv::Mat &initialFrame);
  bool buildCandidateMask(const cv::Mat &frame);
  std::vector<Candidate> extractCandidates();
  bool selectBestCandidate(const std::vector<Candidate> &candidates,
                           Candidate &bestCandidate, float &runnerUpScore);
  void buildSelectedMask(int label);
  bool updateColorModelIfConfident(const Candidate &candidate,
                                   float runnerUpScore);
  bool reacquisitionMode() const;
  void setError(const std::string &error);

  static float wrappedHueDistance(float a, float b);
  static float clamp01(float value);
  static float computeIoU(const cv::Rect &a, const cv::Rect &b);
  static HSVStats computeMaskedHSVStats(const cv::Mat &hsv,
                                        const cv::Mat &mask);

  float lowerRes_;
  int width_ = 0;
  int height_ = 0;
  cv::Size frameSize_;

  cv::Mat hsvFrame_;
  cv::Mat candidateMask_;
  cv::Mat cleanedMask_;
  cv::Mat labelImage_;
  cv::Mat stats_;
  cv::Mat centroids_;
  cv::Mat selectedMaskFrame_;
  cv::Mat erodedMask_;
  cv::Mat latestMask_;
  cv::Mat lastGoodMask_;

  float targetHue_ = 0.0f;
  float targetSat_ = 0.0f;
  float targetVal_ = 0.0f;
  cv::Point2f prevCentroid_{0.0f, 0.0f};
  cv::Rect prevBox_;
  int prevArea_ = 0;
  float confidence_ = 0.0f;
  int lostFrames_ = 0;
  bool haveTrack_ = false;
  bool ready_ = false;
  std::string lastError_;

  static constexpr int kPatchRadius_ = 4;
  static constexpr int kMinBlobArea_ = 120;
  static constexpr int kOpenKernel_ = 3;
  static constexpr int kCloseKernel_ = 5;
  static constexpr int kErodeKernel_ = 5;
  static constexpr int kBaseHueTolerance_ = 10;
  static constexpr int kMaxHueTolerance_ = 20;
  static constexpr int kSatTolerance_ = 70;
  static constexpr int kValTolerance_ = 90;
  static constexpr int kMinSaturation_ = 50;
  static constexpr float kColorUpdateAlpha_ = 0.06f;
  static constexpr float kMinConfidenceForUpdate_ = 0.45f;
  static constexpr float kMinConfidenceForTrack_ = 0.25f;
  static constexpr float kUpdateHueDistanceLimit_ = 18.0f;
  static constexpr float kRunnerUpMargin_ = 0.08f;
  static constexpr int kHoldLastMaskFrames_ = 6;
  static constexpr int kReacquireFrames_ = 18;
};
