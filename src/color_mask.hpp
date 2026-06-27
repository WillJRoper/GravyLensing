/**
 * @file color_mask.hpp
 *
 * Color tracking worker for GravyLensing.
 */

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <QObject>

#include <opencv2/opencv.hpp>

/**
 * @brief Color-tracking worker used as an alternative mask source.
 *
 * ColorMaskWorker operates in one of two submodes, selected via
 * setTrackedBlobMode():
 *
 *   - Fixed Color Key (default):  Pixels within fixed HSV tolerance
 *     of the target colour produce a mask.  This behaves like a chroma-key
 *     matte and is fast, stable, and recommended for most colour-key uses.
 *
 *   - Tracked Color Blob (advanced):  Uses connected-components and
 *     blob-continuity heuristics to track a specific coloured region.
 *     Select this when you need to track a single object rather than
 *     every pixel matching the target colour.
 *
 * In both modes the target is chosen explicitly by the user via
 * Shift+S / File > Select Color..., never automatically.
 */
class ColorMaskWorker : public QObject {
  Q_OBJECT

public:
  /// Build an inactive color tracker that can be initialized later.
  explicit ColorMaskWorker(float lowerRes = 1.0f);

  /// Candidate blob extracted from the thresholded mask.
  struct Candidate {
    int label = 0;
    int area = 0;
    cv::Rect bbox;
    cv::Point2f centroid;
    bool touchesBorder = false;
    float score = 0.0f;
  };

  /// Mean and spread of HSV over a masked region.
  struct HSVStats {
    float hue = 0.0f;
    float sat = 0.0f;
    float val = 0.0f;
    float hueSpread = 0.0f;
    float satSpread = 0.0f;
    float valSpread = 0.0f;
    int count = 0;
  };

  /// Build a color tracker from an initial camera frame and output scale.
  ColorMaskWorker(const cv::Mat &initialFrame, float lowerRes = 1.0f);

  /// Build a tracker pre-loaded with an explicit HSV target (skips the
  /// interactive colour picker).  Intended for tests and scripting.
  ColorMaskWorker(float hue, float sat, float val,
                  int width, int height, float lowerRes = 1.0f);

  /// Whether initialization completed successfully.
  bool isReady() const { return ready_; }

  /// Thread-safe frame submission that coalesces stale work.
  void submitFrame(const cv::Mat &frame);

  /// Whether this worker should process incoming frames.
  bool isEnabled() const { return enabled_; }
  bool trackedBlobMode() const { return trackedBlobMode_; }

  /// Most recent initialization or runtime error message.
  const std::string &lastError() const { return lastError_; }

  /// Current selected target (only valid when isReady() is true).
  float targetHue() const { return targetHue_; }
  float targetSat() const { return targetSat_; }
  float targetVal() const { return targetVal_; }
  int adaptiveHueTolerance() const { return adaptiveHueTol_; }
  int adaptiveSatTolerance() const { return adaptiveSatTol_; }
  int adaptiveValTolerance() const { return adaptiveValTol_; }

  // --- static utilities (public for testing) ------------------------------
  static float wrappedHueDistance(float a, float b);
  static float clamp01(float value);
  static float computeIoU(const cv::Rect &a, const cv::Rect &b);
  static HSVStats computeMaskedHSVStats(const cv::Mat &hsv,
                                         const cv::Mat &mask);

  /// Run the interactive colour picker on the calling thread and return the
  /// HSV statistics of the selected pixel region.  Returns {0,0,0,...} and
  /// count=0 if the user cancelled.
  static HSVStats runInteractiveColorPicker(const cv::Mat &frame);

public Q_SLOTS:
  /// Process a new camera frame and emit a mask if tracking succeeds.
  void onFrame(const cv::Mat &frame);

  /// Enable or disable mask generation while keeping the worker alive.
  void setEnabled(bool enabled);

  /// Switch between Fixed Color Key (false) and Tracked Color Blob (true).
  void setTrackedBlobMode(bool enabled) { trackedBlobMode_ = enabled; }

  /// Override the HSV tolerances used for the keyed colour range.
  void setTolerances(int hue, int sat, int val);

  /// Resize the output mask when the background geometry changes.
  void onBackgroundChange(const cv::Mat &background);

  /// On the next incoming frame, drop the current model and pop up the
  /// interactive colour picker again.  The mask is held frozen until the
  /// user clicks or cancels.
  void triggerReselect();

  /// Called from the main thread after the user completes a colour re-pick.
  void applyReselectionTarget(float hue, float sat, float val, int hueTol,
                              int satTol, int valTol, bool success);

signals:
  /// Emitted when a new binary mask is available for lensing.
  void maskReady(const cv::Mat &mask);

  /// Emitted when color-mode initialization or tracking fails.
  void maskError(const std::string &error);

  /// Emitted after a (re)selection attempt; true means a usable target exists.
  void selectionStateChanged(bool ready);

  /// Emitted from the mask thread when the user has requested a colour
  /// re-pick.  The main thread must open the picker and call
  /// applyReselectionTarget() with the result.
  void reselectionRequested(const cv::Mat &frame);

private:
  void updateGeometry(int width, int height);
  void ensureFrameBuffers(const cv::Size &size);
  bool initializeColorModel(const cv::Mat &initialFrame);
  bool buildCandidateMask(const cv::Mat &frame);
  std::vector<Candidate> extractCandidates();
  bool selectBestCandidate(const std::vector<Candidate> &candidates,
                           Candidate &bestCandidate, float &runnerUpScore);
  void buildSelectedMask(int label);
  bool reacquisitionMode() const;
  void setError(const std::string &error);
  void drainPendingFrame();

  // Output scaling matches the lower-resolution lensing path.
  float lowerRes_;
  int width_ = 0;
  int height_ = 0;
  cv::Size frameSize_;

  // Scratch buffers reused every frame to avoid churn.
  cv::Mat hsvFrame_;
  cv::Mat candidateMask_;
  cv::Mat cleanedMask_;
  cv::Mat labelImage_;
  cv::Mat stats_;
  cv::Mat centroids_;
  cv::Mat selectedMaskFrame_;
  cv::Mat erodedMask_;
  cv::Mat scaledFrame_;
  cv::Mat latestMask_;
  cv::Mat lastGoodMask_;
  cv::Mat smoothedMask_;

  // Running color model and track state.
  float targetHue_ = 0.0f;
  float targetSat_ = 0.0f;
  float targetVal_ = 0.0f;
  int adaptiveHueTol_ = 0;
  int adaptiveSatTol_ = 0;
  int adaptiveValTol_ = 0;
  cv::Point2f prevCentroid_{0.0f, 0.0f};
  cv::Rect prevBox_;
  int prevArea_ = 0;
  int lostFrames_ = 0;
  bool haveTrack_ = false;
  bool enabled_ = false;
  bool ready_ = false;
  bool requestReselect_ = false;
  bool trackedBlobMode_ = false;
  int frameCount_ = 0;
  std::string lastError_;

  // Coalesced frame delivery: keep only the most recent frame while work is
  // in flight so the tracker stays responsive under load.
  std::mutex pendingFrameMutex_;
  cv::Mat pendingFrame_;
  bool pendingFrameDrainScheduled_ = false;

  // Tracking defaults. These are deliberately conservative to prefer stability.
  static constexpr int kPatchRadius_ = 20;
  static constexpr int kMinBlobArea_ = 500;
  static constexpr int kOpenKernel_ = 3;
  static constexpr int kCloseKernel_ = 9;
  static constexpr int kErodeKernel_ = 5;
  static constexpr int kBaseHueTolerance_ = 12;
  static constexpr int kMaxHueTolerance_ = 30;
  static constexpr int kSatTolerance_ = 60;
  static constexpr int kValTolerance_ = 80;
  static constexpr int kMinSaturation_ = 50;
  static constexpr float kMinConfidenceForTrack_ = 0.25f;
  static constexpr int kHoldLastMaskFrames_ = 6;
  static constexpr int kReacquireFrames_ = 18;
  static constexpr float kMaskSmoothAlpha_ = 0.5f;
};
