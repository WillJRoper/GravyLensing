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

  /// Camera access states this process can be in.
  enum class Access { Granted, Denied, Undecided };

  /// Current camera authorization, without prompting.
  static Access accessStatus();

  /// Show the system camera prompt and block until the user answers.
  /// AVFoundation delivers no frames until then, so capture must not be
  /// started before this returns.
  static bool requestAccess();

  bool open(std::string &error, int desiredFps = 30, int desiredWidth = 1280,
            int desiredHeight = 720);
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
