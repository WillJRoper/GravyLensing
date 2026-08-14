#pragma once

#ifdef __APPLE__

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "apple_video_frame.hpp"

class AvFoundationCamera {
public:
  class Impl;

  explicit AvFoundationCamera(int deviceIndex);
  ~AvFoundationCamera();

  static std::vector<std::string> availableDeviceNames();

  bool open(std::string &error, int desiredFps = 30);
  void close();

  bool waitForFrame(cv::Mat &frame, int timeoutMs,
                    const std::atomic<bool> *stopRequested = nullptr);
  bool latestFrame(cv::Mat &frame) const;
  bool waitForNativeFrame(AppleVideoFrame &nativeFrame, int timeoutMs,
                          const std::atomic<bool> *stopRequested = nullptr);
  bool latestNativeFrame(AppleVideoFrame &nativeFrame) const;
  bool waitForFrame(cv::Mat &frame, AppleVideoFrame &nativeFrame, int timeoutMs,
                    const std::atomic<bool> *stopRequested = nullptr);
  bool latestFrame(cv::Mat &frame, AppleVideoFrame &nativeFrame) const;

  double width() const;
  double height() const;
  double fps() const;
  const char *backendName() const;

private:
  std::unique_ptr<Impl> impl_;
};

#endif
