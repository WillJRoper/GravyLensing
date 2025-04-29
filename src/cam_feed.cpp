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

#include "cam_feed.hpp"
#include <atomic>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

/**
 * @brief Constructor for CameraFeed.
 *
 * @param deviceIndex  Camera device index (default = 0)
 *
 * @return true if camera opened successfully
 */
CameraFeed::CameraFeed(int deviceIndex, int width, int height)
    : deviceIndex_(deviceIndex), width_(width), height_(height) {

  // Open the camera
  cap_.open(deviceIndex);

  // Check if the camera opened successfully
  if (!cap_.isOpened()) {
    std::cerr << "Failed to open camera with index " << deviceIndex_ << "\n";
    std::exit(EXIT_FAILURE);
  }

  // Set the camera resolution to match the background (simplifies everything
  // later on)
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
}

/**
 * @brief Capture a frame from the camera.
 *
 * @param frame  Output BGR image
 *
 * @return true if frame read successfully
 */
bool CameraFeed::captureFrame() {

  // Is the camera open?
  if (!cap_.isOpened()) {
    std::cerr << "Camera not open\n";
    return false;
  }

  // Try and grab a new frame with the lock
  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (newFrameReady_) {
      // Already have a new frame, no need to capture again
      return true;
    }

    // Clear the latest frame
    latestFrame_.release();

    // Read a new frame
    if (!cap_.read(latestFrame_)) {
      std::cerr << "Failed to read frame from camera\n";
      return false;
    }

    // Mark the new frame as ready
    newFrameReady_ = true;

    // Mirror it horizontally so on-screen motion matches real motion
    cv::flip(latestFrame_, latestFrame_, /*flipCode=*/1);
  }

  // No we are unlocked and have everything we need the rest of the code can
  // do its thang...

  return true;
}

/**
 * @brief Update the geometry of the camera feed.
 *
 * @param width  New width of the camera feed
 * @param height  New height of the camera feed
 * */
void CameraFeed::updateGeometry(int width, int height) {
  width_ = width;
  height_ = height;
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}
