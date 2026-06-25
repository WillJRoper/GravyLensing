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
#ifdef __APPLE__
#include "avfoundation_camera.hpp"
#endif
#include "cam_feed.hpp"
#include "perf_log.hpp"

/**
 * @brief Display an interactive ROI selector and build a mask.
 *
 * Uses a mouse-callback-based click-and-drag selector instead of
 * cv::selectROI, which is unreliable alongside Qt on macOS (key
 * presses can land in the Qt window instead of the OpenCV window).
 *
 * @param frame The image on which to select the ROI (modified for preview).
 * @param flip If true, the preview will be mirrored horizontally.
 * @return A pair consisting of the selected cv::Rect and its binary mask.
 */
std::pair<cv::Rect, cv::Mat> selectROIAndMask(cv::Mat &frame, bool flip) {

  if (flip)
    cv::flip(frame, frame, 1);

  struct State {
    cv::Point start{-1, -1};
    cv::Point end{-1, -1};
    bool drawing = false;
    bool confirmed = false;
    bool cancelled = false;
  } state;

  auto onMouse = [](int event, int x, int y, int, void *data) {
    auto *s = static_cast<State *>(data);
    if (event == cv::EVENT_LBUTTONDOWN) {
      s->start = cv::Point(x, y);
      s->end = cv::Point(x, y);
      s->drawing = true;
    } else if (event == cv::EVENT_MOUSEMOVE && s->drawing) {
      s->end = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP) {
      s->end = cv::Point(x, y);
      s->drawing = false;
    }
  };

  const std::string winName = "Select ROI";
  cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(winName, onMouse, &state);

  cv::Mat preview;
  while (!state.confirmed && !state.cancelled) {
    preview = frame.clone();
    const char *instr =
        "Drag to select ROI.  ENTER=confirm  ESC=cancel";
    cv::putText(preview, instr, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    if (state.start.x >= 0 && state.end.x >= 0) {
      cv::Rect r(state.start, state.end);
      if (r.width > 0 || r.height > 0)
        cv::rectangle(preview, r, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow(winName, preview);
    int key = cv::waitKey(30);
    if (key == 13 || key == 32)       // Enter or Space
      state.confirmed = true;
    else if (key == 27 || key == 'c' || key == 'C')
      state.cancelled = true;
  }

  cv::destroyWindow(winName);

  cv::Rect rect;
  cv::Mat mask;
  if (state.confirmed && state.start.x >= 0 && state.end.x >= 0) {
    rect = cv::Rect(state.start, state.end);
    if (rect.width == 0 && rect.height == 0) {
      // Single-click (no drag) — use full frame
      rect = cv::Rect(0, 0, frame.cols, frame.rows);
    }
  } else {
    rect = cv::Rect(0, 0, frame.cols, frame.rows);
  }

  mask = cv::Mat::zeros(frame.size(), CV_8UC1);
  cv::rectangle(mask, rect, cv::Scalar(255), cv::FILLED);
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
 * @param flip Whether to flip the camera feed horizontally (default is false).
 * @param selectROI Whether to allow the user to select a region of interest
 *  (ROI) in the camera feed (default is false).
 */
CameraFeed::CameraFeed(int deviceIndex, bool flip, bool selectROI)
    : deviceIndex_(deviceIndex), flip_(flip), doingROI_(selectROI) {

  // Initialize the camera feed and ensure it is opened successfully
  if (!initCamera()) {
    emit captureError("Failed to open camera device " +
                      std::to_string(deviceIndex_));
    isOpen_ = false;
    return;
  }

  // Ensure we have opened a valid camera with a valid index
#ifdef __APPLE__
  if (!avCamera_) {
#else
  if (!cap_.isOpened()) {
#endif
    emit captureError("Failed to open camera device " +
                      std::to_string(deviceIndex_));
    isOpen_ = false;
    return;
  }

  // We are now open
  isOpen_ = true;

  // Check we are getting frames at all
  cv::Mat testFrame;
  if (!readFrame(testFrame, true) || testFrame.empty()) {
    emit captureError("Failed to read initial frame from camera device " +
                      std::to_string(deviceIndex_));
    return;
  }

  std::cout << "[CameraFeed] Camera " << deviceIndex_
            << " opened successfully.\n";
  std::cout << "[CameraFeed] Camera resolution: "
#ifdef __APPLE__
            << avCamera_->width() << "x" << avCamera_->height() << "\n";
  std::cout << "[CameraFeed] Camera FPS target: " << avCamera_->fps() << "\n";
  std::cout << "[CameraFeed] Camera backend: " << avCamera_->backendName() << "\n";
#else
            << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
  std::cout << "[CameraFeed] Camera FPS: " << cap_.get(cv::CAP_PROP_FPS)
            << "\n";
  std::cout << "[CameraFeed] Camera fourcc: " << cap_.get(cv::CAP_PROP_FOURCC)
            << "\n";
  std::cout << "[CameraFeed] Camera backend: " << cap_.get(cv::CAP_PROP_BACKEND)
            << "\n";
#endif

  // If we aren't selecting an ROI we are done and can move on
  if (!doingROI_) {
    return;
  }

  // Grab frame for ROI selection
  cv::Mat firstFrame;
  if (!readFrame(firstFrame, true) || firstFrame.empty()) {
    emit captureError("Failed to grab initial frame for ROI selection");
    return;
  }

  // Delegate to static helper
  std::tie(roiRect_, roiMask_) = selectROIAndMask(firstFrame, flip_);

  std::cout << "[CameraFeed] ROI: (x, y)=(" << roiRect_.x << ", " << roiRect_.y
            << ") " << "widthxheight=" << roiRect_.width << "x"
            << roiRect_.height << "\n";
}

/**
 * @brief Destructor for the CameraFeed class.
 *
 * This destructor releases the camera feed if it is opened.
 */
CameraFeed::~CameraFeed() {
#ifdef __APPLE__
  if (avCamera_) {
    avCamera_->close();
  }
#endif
  if (cap_.isOpened())
    cap_.release();
}

void CameraFeed::setROI(cv::Rect rect, cv::Mat mask) {
  std::lock_guard<std::mutex> lock(roiMutex_);
  roiRect_ = rect;
  roiMask_ = mask.clone();
  doingROI_.store(true);
  std::cout << "[CameraFeed] ROI updated: "
            << roiRect_.width << "x" << roiRect_.height << "\n";
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
#ifdef __APPLE__
  avCamera_ = std::make_unique<AvFoundationCamera>(deviceIndex_);
  std::string error;
  if (!avCamera_->open(error)) {
    std::cerr << "[CameraFeed] AVFoundation open failed: " << error << "\n";
    avCamera_.reset();
    return false;
  }
  return true;
#else

  // Check if the camera is already opened
  if (cap_.isOpened())
    cap_.release();

  // Open the camera device. On macOS, prefer AVFoundation explicitly so we
  // avoid backend ambiguity and can tune the capture path for lower latency.
#ifdef __APPLE__
  cap_.open(deviceIndex_, cv::CAP_AVFOUNDATION);
#else
  cap_.open(deviceIndex_);
#endif
  if (!cap_.isOpened())
    return false;

  // Ask OpenCV to keep as little camera-side buffering as the backend allows.
  // Unsupported properties are ignored by OpenCV/backends.
  cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

  // Discard the first several frames so auto-exposure can settle before
  // we read a frame used for ROI/color selection.
  cv::Mat warmup;
  for (int i = 0; i < 15; ++i) {
    cap_ >> warmup;
  }

  // Lock auto-exposure so the camera stops re-metering every time a bright
  // OpenCV or Qt window appears on the monitor.  Leave auto-white-balance
  // alone — locking it at the hardware default (often 0 K) destroys colour
  // fidelity and makes colour tracking impossible.
  if (!cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.0)) {
    // Some backends (e.g. V4L2) interpret 0.25 = manual, 0.75 = auto.
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
  }

  std::cout << "[CameraFeed] Exposure locked at "
            << cap_.get(cv::CAP_PROP_EXPOSURE) << "\n";
  std::cout << "[CameraFeed] Active backend: "
            << cap_.get(cv::CAP_PROP_BACKEND) << "\n";

  return true;
#endif
}

bool CameraFeed::readFrame(cv::Mat &frame, bool latestOnly) {
#ifdef __APPLE__
  if (!avCamera_) {
    return false;
  }
  if (latestOnly) {
    return avCamera_->latestFrame(frame);
  }
  if (avCamera_->waitForFrame(frame, 500, &stopRequested_)) {
    return true;
  }
  return !stopRequested_.load() && avCamera_->latestFrame(frame);
#else
  if (latestOnly) {
    return cap_.read(frame) && !frame.empty();
  }
  return cap_.grab() && cap_.retrieve(frame) && !frame.empty();
#endif
}

/**
 * @brief Start the camera capture loop.
 *
 * This function starts a loop that captures frames from the camera feed and
 * emits the captured frames.
 */
void CameraFeed::startCaptureLoop() {

  stopRequested_.store(false);
  static thread_local PerfLog perf("capture", 120);

  // Define a local reusable header for the frame
  cv::Mat frame;

  // Loop until the end of time (or until the thread is stopped)
  while (!stopRequested_.load() && !QCoreApplication::closingDown()) {
    const auto t0 = std::chrono::steady_clock::now();

    // Capture a frame from the camera
    if (!readFrame(frame)) {
      emit captureError("Frame capture failed");
      std::cout << "[CameraFeed] Frame capture failed\n";
      break;
    }

    // Flip the frame horizontally if requested
    if (flip_) {
      cv::flip(frame, frame, 1);
    }

    // Apply ROI mask, crop to ROI and emit if we are doing ROI selection,
    // otherwise just emit the full frame
    if (doingROI_.load()) {
      cv::Rect rect;
      cv::Mat mask;
      {
        std::lock_guard<std::mutex> lock(roiMutex_);
        rect = roiRect_;
        mask = roiMask_;
      }
      emit frameCaptured(
          applyROIMaskAndCrop(frame, mask, rect).clone());
    } else {
      emit frameCaptured(frame.clone());
    }

    const auto t1 = std::chrono::steady_clock::now();
    perf.addSample(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
}

void CameraFeed::stopCaptureLoop() { stopRequested_.store(true); }

cv::Mat CameraFeed::captureSetupFrame() {
  cv::Mat frame;
  if (!readFrame(frame, true) || frame.empty()) {
    emit captureError("Failed to capture setup frame");
    return cv::Mat();
  }

  if (flip_) {
    cv::flip(frame, frame, 1);
  }

  if (doingROI_.load()) {
    cv::Rect rect;
    cv::Mat mask;
    {
      std::lock_guard<std::mutex> lock(roiMutex_);
      rect = roiRect_;
      mask = roiMask_;
    }
    return applyROIMaskAndCrop(frame, mask, rect);
  }

  return frame;
}

cv::Mat CameraFeed::captureSelectionFrame() {
  cv::Mat frame;
  if (!readFrame(frame, true) || frame.empty()) {
    emit captureError("Failed to capture selection frame");
    return cv::Mat();
  }

  if (flip_) {
    cv::flip(frame, frame, 1);
  }

  return frame;
}
