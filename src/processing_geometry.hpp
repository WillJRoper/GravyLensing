#pragma once

#include <algorithm>
#include <cmath>

#include <opencv2/core/types.hpp>

inline cv::Size calculationSize(const cv::Size &outputSize, float scale) {
  return {std::max(1, static_cast<int>(std::lround(outputSize.width * scale))),
          std::max(1,
                   static_cast<int>(std::lround(outputSize.height * scale)))};
}
