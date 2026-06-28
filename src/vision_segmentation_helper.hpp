#pragma once

#ifdef __APPLE__

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "apple_video_frame.hpp"
#include "segmentation_worker.hpp"

class SegmentationWorker::ApplePersonSegmentationHelper {
public:
  ApplePersonSegmentationHelper();
  ~ApplePersonSegmentationHelper();

  bool isAvailable() const;
  bool generatePersonProbability(const cv::Mat &frame, int targetWidth,
                                 int targetHeight, cv::Mat &outProb,
                                 std::string &error);
  bool generatePersonProbability(const AppleVideoFrame &frame, int targetWidth,
                                 int targetHeight, cv::Mat &outProb,
                                 std::string &error);
  bool generatePersonProbability(const AppleVideoFrame &frame,
                                 const cv::Rect2f &normalizedCrop,
                                 int targetWidth, int targetHeight,
                                 cv::Mat &outProb, std::string &error);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#endif
