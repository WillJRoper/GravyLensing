#pragma once

#include <opencv2/imgproc.hpp>

inline cv::Mat compositeMaskedForeground(const cv::Mat &background,
                                         const cv::Mat &foreground,
                                         const cv::Mat &mask) {
  if (background.empty() || foreground.empty() || mask.empty())
    return background.clone();

  cv::Mat resizedForeground;
  cv::Mat resizedMask;
  cv::resize(foreground, resizedForeground, background.size(), 0, 0,
             cv::INTER_LINEAR);
  cv::resize(mask, resizedMask, background.size(), 0, 0, cv::INTER_NEAREST);
  cv::threshold(resizedMask, resizedMask, 127, 255, cv::THRESH_BINARY);
  cv::Mat composite = background.clone();
  resizedForeground.copyTo(composite, resizedMask);
  return composite;
}
