/**
 * @file cam_feed.hpp
 *
 * Camera feed extension for Gravy Lensing
 *
 * This extension captures frames from a camera feed using openCV.
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

// Standard includes
#include <atomic>
#include <mutex>
#include <string>
#include <memory>
#include <utility>

// Qt includes
#include <QObject>

// External includes
#include <opencv2/opencv.hpp>

#ifdef __APPLE__
#include "apple_video_frame.hpp"
#endif

#ifdef __APPLE__
class AvFoundationCamera;
#endif

/**
 * @brief CameraFeed class
 *
 * This class captures frames from a camera feed using openCV.
 *
 * It emits signals when a new frame is captured or if there is an error
 * opening or reading the camera.
 */
class CameraFeed : public QObject {
  Q_OBJECT

public:
  CameraFeed(int deviceIndex = 0, bool flip = false, bool selectROI = false);
  ~CameraFeed();

  /// Start continuous capture in this thread
  Q_INVOKABLE void startCaptureLoop();

  /// Request that the capture loop exits (non-blocking; returns immediately).
  /// Callers must coordinate the remainder of the shutdown sequence —
  /// disconnect signals, shut down downstream workers, then quit/wait
  /// the camera thread — exactly as documented in stopPipeline().
  void stopCaptureLoop();

  // Capture a setup frame using the same transforms as runtime output.
  /// Returns a frame with the same flip/ROI transforms applied as the
  /// live capture loop.
  Q_INVOKABLE cv::Mat captureSetupFrame();

  /// Capture a full frame ignoring any active ROI crop — used by region
  /// selection so the user can draw a new ROI from the unfiltered feed.
  cv::Mat captureSelectionFrame();

  /// Apply a new ROI rectangle and mask at runtime.
  Q_INVOKABLE void setROI(cv::Rect rect, cv::Mat mask);

  // Is the camera open?
  bool isOpen() const { return isOpen_; }

  /// Query the current ROI state so it can be preserved across restarts.
  bool hasROI() const { return doingROI_.load(); }
  cv::Rect roiRect() const {
    std::lock_guard<std::mutex> lock(roiMutex_);
    return roiRect_;
  }
  cv::Mat roiMask() const {
    std::lock_guard<std::mutex> lock(roiMutex_);
    return roiMask_.clone();
  }

signals:
  /// Emitted as soon as a new frame is ready
  void frameCaptured(const cv::Mat &frame);

#ifdef __APPLE__
  void nativeFrameCaptured(const AppleVideoFrame &frame);
#endif

  /// Emitted if there's an error opening or reading the camera
  void captureError(const std::string &error);

private:
  bool initCamera(); ///< Called by ctor to open cap_
  bool readFrame(cv::Mat &frame, bool latestOnly = false);

  // The device index for the camera (0 for default camera)
  int deviceIndex_;

  // OpenCV video capture object
  cv::VideoCapture cap_;

#ifdef __APPLE__
  std::unique_ptr<AvFoundationCamera> avCamera_;
#endif

  // ROI selection and mask — protected by roiMutex_ when accessed from
  // outside the capture thread.
  mutable std::mutex roiMutex_;
  cv::Rect roiRect_;
  cv::Mat roiMask_;

  // Are we flipping the camera feed horizontally?
  bool flip_ = false;

  // Are we doing ROI selection? (atomic — read from capture thread,
  // written from main thread)
  std::atomic<bool> doingROI_{false};

  // Did we open ok?
  bool isOpen_ = false;

  // Cooperative stop flag for the capture loop.
  std::atomic<bool> stopRequested_{false};
};

/// Interactive ROI selector (usable from the main thread).
std::pair<cv::Rect, cv::Mat> selectROIAndMask(cv::Mat &frame, bool flip);
