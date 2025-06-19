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

// Qt includes
#include <QObject>

// External includes
#include <opencv2/opencv.hpp>

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
  CameraFeed(int deviceIndex = 0, bool flip = false);
  ~CameraFeed();

  /// Start continuous capture in this thread
  Q_INVOKABLE void startCaptureLoop();

signals:
  /// Emitted as soon as a new frame is ready
  void frameCaptured(const cv::Mat &frame);

  /// Emitted if there's an error opening or reading the camera
  void captureError(const QString &msg);

private:
  bool initCamera(); ///< Called by ctor to open cap_

  // The device index for the camera (0 for default camera)
  int deviceIndex_;

  // OpenCV video capture object
  cv::VideoCapture cap_;

  // ROI selection and mask
  cv::Rect roiRect_;
  cv::Mat roiMask_;

  // Are we flipping the camera feed horizontally?
  bool flip_ = false;
};
