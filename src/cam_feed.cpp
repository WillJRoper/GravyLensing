/**
 * @file cam_feed.cpp
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

// Standard includes
#include <iostream>
#include <stdlib.h>

// Qt includes
#include <QCoreApplication>
#include <QThread>

// External includes
#include <opencv2/opencv.hpp>

// Local includes
#include "cam_feed.hpp"

/**
 * @brief Constructor for the CameraFeed class.
 *
 * This constructor initializes the camera feed with the specified device index,
 * width, and height.
 *
 * @param deviceIndex The index of the camera device (default is 0).
 */
CameraFeed::CameraFeed(int deviceIndex) : deviceIndex_(deviceIndex) {

  // Initialize the camera feed and ensure it is opened successfully
  if (!initCamera()) {
    emit captureError("Failed to open camera " + QString::number(deviceIndex_));
  }

  std::cout << "[CameraFeed] Camera " << deviceIndex_
            << " opened successfully.\n";
  std::cout << "[CameraFeed] Camera resolution: "
            << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
  std::cout << "[CameraFeed] Camera FPS: " << cap_.get(cv::CAP_PROP_FPS)
            << "\n";
  std::cout << "[CameraFeed] Camera fourcc: " << cap_.get(cv::CAP_PROP_FOURCC)
            << "\n";
  std::cout << "[CameraFeed] Camera backend: " << cap_.get(cv::CAP_PROP_BACKEND)
            << "\n";
}

/**
 * @brief Destructor for the CameraFeed class.
 *
 * This destructor releases the camera feed if it is opened.
 */
CameraFeed::~CameraFeed() {
  if (cap_.isOpened())
    cap_.release();
}

/**
 * @brief Initialize the camera feed.
 *
 * This function opens the camera device and sets the width and height of the
 * camera feed.
 *
 * @return true if the camera is opened successfully, false otherwise.
 */
bool CameraFeed::initCamera() {

  // Check if the camera is already opened
  if (cap_.isOpened())
    cap_.release();

  // Open the camera device
  cap_.open(deviceIndex_);
  if (!cap_.isOpened())
    return false;

  return true;
}

/**
 * @brief Start the camera capture loop.
 *
 * This function starts a loop that captures frames from the camera feed and
 * emits the captured frames.
 */
void CameraFeed::startCaptureLoop() {

  // Define a local reusable header for the frame
  cv::Mat frame;

  // Loop until the end of time (or until the thread is stopped)
  while (QThread::currentThread()->isRunning() &&
         !QCoreApplication::closingDown()) {

    // Capture a frame from the camera
    if (!cap_.grab() || !cap_.retrieve(frame)) {
      emit captureError("Frame capture failed");
      break;
    }

    // Flip the frame horizontally to ensure the correct orientation
    cv::flip(frame, frame, /*flipCode=*/1);

    // Emit the captured frame
    emit frameCaptured(frame);
  }
}
