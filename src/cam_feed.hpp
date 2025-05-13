/**
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

#include <atomic>
#include <iostream>
#include <opencv2/opencv.hpp>

class CameraFeed {

public:
  // Define a container for the latest frame
  cv::Mat latestFrame_;

  // Geometry of the camera feed
  int width_;
  int height_;

  // Constructor
  CameraFeed(int deviceIndex = 0, int width = 640, int height = 480);

  // Destructor
  ~CameraFeed();

  /// Capture a frame from the camera
  bool captureFrame();

  // Update the geometry of the camera feed
  void updateGeometry(int width, int height);

private:
  /// Camera device index
  int deviceIndex_;

  // OpenCV video capture object
  cv::VideoCapture cap_;
};
