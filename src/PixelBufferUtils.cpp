#import <Foundation/Foundation.h>

#include "PixelBufferUtils.hpp"

CVPixelBufferRef matToPixelBuffer(const cv::Mat &mat) {
  CVPixelBufferRef buffer = nullptr;
  NSDictionary *attrs = @{
    (id)kCVPixelBufferCGImageCompatibilityKey : @YES,
    (id)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES
  };
  CVPixelBufferCreate(kCFAllocatorDefault, mat.cols, mat.rows,
                      kCVPixelFormatType_32BGRA,
                      (__bridge CFDictionaryRef)attrs, &buffer);
  CVPixelBufferLockBaseAddress(buffer, 0);
  void *dest = CVPixelBufferGetBaseAddress(buffer);
  size_t rowBytes = CVPixelBufferGetBytesPerRow(buffer);
  // assume mat.type()==CV_8UC4 (BGRA)
  for (int y = 0; y < mat.rows; ++y) {
    memcpy((char *)dest + y * rowBytes, mat.ptr(y), mat.cols * 4);
  }
  CVPixelBufferUnlockBaseAddress(buffer, 0);
  return buffer;
}

cv::Mat pixelBufferToMat(CVPixelBufferRef buf) {
  CVPixelBufferLockBaseAddress(buf, kCVPixelBufferLock_ReadOnly);
  int w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf);
  size_t rowBytes = CVPixelBufferGetBytesPerRow(buf);
  void *base = CVPixelBufferGetBaseAddress(buf);
  // single‐channel 8‐bit mask
  cv::Mat m(h, w, CV_8UC1, base, rowBytes);
  cv::Mat copy = m.clone(); // detach from underlying buffer
  CVPixelBufferUnlockBaseAddress(buf, kCVPixelBufferLock_ReadOnly);
  return copy;
}
