#pragma once

// Undefine Objective-C’s NO so OpenCV enums compile cleanly
#undef NO

#include <CoreVideo/CoreVideo.h>
#include <opencv2/opencv.hpp>

/// Convert a CV_8UC4 (BGRA) cv::Mat → CVPixelBufferRef (32BGRA)
CVPixelBufferRef matToPixelBuffer(const cv::Mat &mat);

/// Wrap a single-channel CVPixelBufferRef → cv::Mat CV_8UC1
cv::Mat pixelBufferToMat(CVPixelBufferRef buf);
