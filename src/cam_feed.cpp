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
 * @brief Display an interactive ROI selector and build a mask.
 *
 * This static helper opens a window to let the user draw a rectangular ROI
 * on the provided frame. If the user cancels, the full frame is used.
 * It returns both the selected rectangle and a binary mask of the same size.
 *
 * @param frame The image on which to select the ROI (modified for preview).
 * @return A pair consisting of the selected cv::Rect and its binary mask as
 * cv::Mat.
 */
static std::pair<cv::Rect, cv::Mat> selectROIAndMask(cv::Mat &frame) {
  // Mirror preview
  // cv::flip(frame, frame, 1);

  // Show selector
  cv::namedWindow("Select ROI", cv::WINDOW_AUTOSIZE);
  cv::Rect sel = cv::selectROI("Select ROI", frame);
  cv::destroyWindow("Select ROI");

  // Record the selection and store the mask
  cv::Rect rect;
  cv::Mat mask;
  if (sel.width > 0 && sel.height > 0) {
    rect = sel;
    mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::rectangle(mask, sel, cv::Scalar(255), cv::FILLED);
  } else {
    // Full-frame fallback
    rect = cv::Rect(0, 0, frame.cols, frame.rows);
    mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(255));
  }

  // Return the selected rectangle and mask
  return {rect, mask};
}

/**
 * @brief Crop an image to the given ROI and apply mask.
 *
 * Extracts the rectangular region defined by roiRect from src,
 * zeroing out any pixels outside the binary mask within that region.
 * The returned image has dimensions roiRect.size().
 *
 * @param src The source image (multi-channel).
 * @param mask The full-frame binary mask; non-zero pixels define valid ROI.
 * @param roiRect The rectangle to crop from src and mask.
 * @return A new cv::Mat of size roiRect.size() containing only the ROI.
 */
static cv::Mat applyROIMaskAndCrop(const cv::Mat &src, const cv::Mat &mask,
                                   const cv::Rect &roiRect) {
  // Crop both source and mask to the rectangle
  cv::Mat srcCrop = src(roiRect);
  cv::Mat maskCrop = mask(roiRect);
  // Allocate output and apply mask
  cv::Mat output = cv::Mat::zeros(srcCrop.size(), srcCrop.type());
  srcCrop.copyTo(output, maskCrop);
  return output;
}

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

  // Grab frame for ROI selection
  cv::Mat firstFrame;
  if (!cap_.read(firstFrame) || firstFrame.empty()) {
    emit captureError("Failed to grab initial frame for ROI selection");
    return;
  }

  // Delegate to static helper
  std::tie(roiRect_, roiMask_) = selectROIAndMask(firstFrame);

  std::cout << "[CameraFeed] ROI: (x, y)=(" << roiRect_.x << ", " << roiRect_.y
            << ") " << "widthxheight=" << roiRect_.width << "x"
            << roiRect_.height << "\n";

  // Report some info about the ROI mask
  std::cout << "[CameraFeed] ROI mask size: " << roiMask_.size()
            << ", type: " << roiMask_.type() << "\n";
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
  cv::Mat flipped;
  cv::Mat masked;

  // Loop until the end of time (or until the thread is stopped)
  while (QThread::currentThread()->isRunning() &&
         !QCoreApplication::closingDown()) {

    // Capture a frame from the camera
    if (!cap_.grab() || !cap_.retrieve(frame)) {
      emit captureError("Frame capture failed");
      std::cout << "[CameraFeed] Frame capture failed\n";
      break;
    }

    // Flip the frame horizontally to ensure the correct orientation
    // cv::flip(frame, flipped, 1);

    // Apply ROI mask and crop to ROI
    masked = applyROIMaskAndCrop(frame, roiMask_, roiRect_);

    // Emit the captured frame
    emit frameCaptured(masked);
  }
}
