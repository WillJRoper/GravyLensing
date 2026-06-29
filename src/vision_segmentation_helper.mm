#ifdef __APPLE__

#include "vision_segmentation_helper.hpp"

#include <chrono>

#include "perf_log.hpp"

#import <CoreVideo/CoreVideo.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <Vision/Vision.h>

namespace {

double elapsedMs(const std::chrono::steady_clock::time_point &start,
                 const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

CVPixelBufferRef createPixelBufferFromBgr(const cv::Mat &frame) {
  if (frame.empty() || frame.type() != CV_8UC3) {
    return nullptr;
  }

  CVPixelBufferRef pixelBuffer = nullptr;
  const NSDictionary *attributes = @{
    (id)kCVPixelBufferCGImageCompatibilityKey : @YES,
    (id)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
  };

  const CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, frame.cols, frame.rows, kCVPixelFormatType_32BGRA,
      (__bridge CFDictionaryRef)attributes, &pixelBuffer);
  if (status != kCVReturnSuccess || pixelBuffer == nullptr) {
    return nullptr;
  }

  CVPixelBufferLockBaseAddress(pixelBuffer, 0);
  auto *dstBase = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(pixelBuffer));
  const size_t dstStride = CVPixelBufferGetBytesPerRow(pixelBuffer);

  for (int y = 0; y < frame.rows; ++y) {
    const auto *srcRow = frame.ptr<cv::Vec3b>(y);
    auto *dstRow = dstBase + static_cast<size_t>(y) * dstStride;
    for (int x = 0; x < frame.cols; ++x) {
      const cv::Vec3b &pixel = srcRow[x];
      const int idx = x * 4;
      dstRow[idx + 0] = pixel[0];
      dstRow[idx + 1] = pixel[1];
      dstRow[idx + 2] = pixel[2];
      dstRow[idx + 3] = 255;
    }
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
  return pixelBuffer;
}

CVPixelBufferRef createBgraPixelBuffer(int width, int height) {
  CVPixelBufferRef pixelBuffer = nullptr;
  const NSDictionary *attributes = @{
    (id)kCVPixelBufferCGImageCompatibilityKey : @YES,
    (id)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
  };
  const CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      (__bridge CFDictionaryRef)attributes, &pixelBuffer);
  if (status != kCVReturnSuccess) {
    return nullptr;
  }
  return pixelBuffer;
}

bool copyMaskToProbabilityMat(CVPixelBufferRef maskBuffer, int targetWidth,
                              int targetHeight, cv::Mat &outProb,
                              std::string &error) {
  if (maskBuffer == nullptr) {
    error = "Vision request returned an empty mask buffer";
    return false;
  }

  CVPixelBufferLockBaseAddress(maskBuffer, kCVPixelBufferLock_ReadOnly);

  const OSType pixelFormat = CVPixelBufferGetPixelFormatType(maskBuffer);
  const int width = static_cast<int>(CVPixelBufferGetWidth(maskBuffer));
  const int height = static_cast<int>(CVPixelBufferGetHeight(maskBuffer));
  const size_t stride = CVPixelBufferGetBytesPerRow(maskBuffer);
  void *baseAddress = CVPixelBufferGetBaseAddress(maskBuffer);

  cv::Mat rawMask;
  if (pixelFormat == kCVPixelFormatType_OneComponent8) {
    cv::Mat mask8(height, width, CV_8UC1, baseAddress, stride);
    mask8.convertTo(rawMask, CV_32F, 1.0 / 255.0);
  } else if (pixelFormat == kCVPixelFormatType_OneComponent16Half) {
    cv::Mat mask16(height, width, CV_16FC1, baseAddress, stride);
    mask16.convertTo(rawMask, CV_32F);
  } else if (pixelFormat == kCVPixelFormatType_OneComponent32Float) {
    rawMask = cv::Mat(height, width, CV_32FC1, baseAddress, stride).clone();
  } else {
    CVPixelBufferUnlockBaseAddress(maskBuffer, kCVPixelBufferLock_ReadOnly);
    error = "Unsupported Vision mask pixel format";
    return false;
  }

  CVPixelBufferUnlockBaseAddress(maskBuffer, kCVPixelBufferLock_ReadOnly);

  if (rawMask.empty()) {
    error = "Vision mask conversion produced an empty probability map";
    return false;
  }

  if (rawMask.cols != targetWidth || rawMask.rows != targetHeight) {
    cv::resize(rawMask, outProb, cv::Size(targetWidth, targetHeight), 0, 0,
               cv::INTER_LINEAR);
  } else {
    outProb = rawMask;
  }

  return true;
}

} // namespace

class SegmentationWorker::ApplePersonSegmentationHelper::Impl {
public:
  Impl() {
    if (@available(macOS 11.0, *)) {
      request_ = [[VNGeneratePersonSegmentationRequest alloc] init];
      request_.qualityLevel = VNGeneratePersonSegmentationRequestQualityLevelBalanced;
      request_.outputPixelFormat = kCVPixelFormatType_OneComponent8;
      ciContext_ = [CIContext contextWithOptions:@{}];
      available_ = true;
    }
  }

  ~Impl() {
    if (scaledBuffer_ != nullptr) {
      CVPixelBufferRelease(scaledBuffer_);
      scaledBuffer_ = nullptr;
    }
  }

  CVPixelBufferRef scaledBufferForSize(int width, int height) {
    if (scaledBuffer_ != nullptr && scaledWidth_ == width &&
        scaledHeight_ == height) {
      return scaledBuffer_;
    }

    if (scaledBuffer_ != nullptr) {
      CVPixelBufferRelease(scaledBuffer_);
      scaledBuffer_ = nullptr;
    }

    scaledBuffer_ = createBgraPixelBuffer(width, height);
    if (scaledBuffer_ != nullptr) {
      scaledWidth_ = width;
      scaledHeight_ = height;
    } else {
      scaledWidth_ = 0;
      scaledHeight_ = 0;
    }
    return scaledBuffer_;
  }

  bool available_ = false;
  VNGeneratePersonSegmentationRequest *request_ = nil;
  CIContext *ciContext_ = nil;
  CVPixelBufferRef scaledBuffer_ = nullptr;
  int scaledWidth_ = 0;
  int scaledHeight_ = 0;
};

SegmentationWorker::ApplePersonSegmentationHelper::ApplePersonSegmentationHelper()
    : impl_(std::make_unique<Impl>()) {}

SegmentationWorker::ApplePersonSegmentationHelper::~ApplePersonSegmentationHelper() = default;

bool SegmentationWorker::ApplePersonSegmentationHelper::isAvailable() const {
  return impl_ != nullptr && impl_->available_;
}

bool SegmentationWorker::ApplePersonSegmentationHelper::generatePersonProbability(
    const cv::Mat &frame, int targetWidth, int targetHeight, cv::Mat &outProb,
    std::string &error) {
  if (!isAvailable()) {
    error = "Vision person segmentation is unavailable on this macOS version";
    return false;
  }
  if (impl_->request_ == nil) {
    error = "Vision request is unavailable";
    return false;
  }

  cv::Mat preparedFrame;
  if (frame.cols != targetWidth || frame.rows != targetHeight) {
    cv::resize(frame, preparedFrame, cv::Size(targetWidth, targetHeight), 0, 0,
               cv::INTER_LINEAR);
  } else {
    preparedFrame = frame;
  }

  CVPixelBufferRef inputBuffer = createPixelBufferFromBgr(preparedFrame);
  if (inputBuffer == nullptr) {
    error = "Failed to create CVPixelBuffer for Vision request";
    return false;
  }

  @autoreleasepool {
    @try {
      NSError *nsError = nil;
      VNImageRequestHandler *handler =
          [[VNImageRequestHandler alloc] initWithCVPixelBuffer:inputBuffer
                                                   orientation:kCGImagePropertyOrientationUp
                                                       options:@{}];

      const BOOL success = [handler performRequests:@[ impl_->request_ ]
                                              error:&nsError];
      if (!success) {
        error = nsError != nil ? nsError.localizedDescription.UTF8String
                               : "Vision request failed";
        CVPixelBufferRelease(inputBuffer);
        return false;
      }

      VNPixelBufferObservation *result = impl_->request_.results.firstObject;
      if (result == nil) {
        error = "Vision request returned no segmentation result";
        CVPixelBufferRelease(inputBuffer);
        return false;
      }

      const bool copied = copyMaskToProbabilityMat(result.pixelBuffer, targetWidth,
                                                    targetHeight, outProb, error);
      CVPixelBufferRelease(inputBuffer);
      return copied;
    } @catch (NSException *exception) {
      CVPixelBufferRelease(inputBuffer);
      error = std::string("Vision pipeline ObjC exception: ") +
              exception.name.UTF8String + " - " +
              exception.reason.UTF8String;
      return false;
    }
  }
}

bool SegmentationWorker::ApplePersonSegmentationHelper::generatePersonProbability(
    const AppleVideoFrame &frame, int targetWidth, int targetHeight,
    cv::Mat &outProb, std::string &error) {
  return generatePersonProbability(frame, cv::Rect2f(), targetWidth,
                                   targetHeight, outProb, error);
}

bool SegmentationWorker::ApplePersonSegmentationHelper::generatePersonProbability(
    const AppleVideoFrame &frame, const cv::Rect2f &normalizedCrop,
    int targetWidth, int targetHeight, cv::Mat &outProb, std::string &error) {
  static thread_local PerfLog scalePerf("person-mask-vision-scale", 60);
  static thread_local PerfLog requestPerf("person-mask-vision-request", 60);
  static thread_local PerfLog copyPerf("person-mask-vision-copy", 60);

  if (!isAvailable()) {
    error = "Vision person segmentation is unavailable on this macOS version";
    return false;
  }
  if (!frame.isValid()) {
    error = "Native Apple video frame is invalid";
    return false;
  }
  if (impl_->ciContext_ == nil) {
    error = "CoreImage context is unavailable";
    return false;
  }
  if (impl_->request_ == nil) {
    error = "Vision request is unavailable";
    return false;
  }

  @autoreleasepool {
    @try {
      CVPixelBufferRef scaledBuffer =
          impl_->scaledBufferForSize(targetWidth, targetHeight);
      if (scaledBuffer == nullptr) {
        error = "Failed to allocate scaled CVPixelBuffer for Vision request";
        return false;
      }

      auto stageStart = std::chrono::steady_clock::now();
      CIImage *sourceImage = [CIImage imageWithCVPixelBuffer:frame.pixelBuffer];
      if (sourceImage == nil) {
        error = "Failed to create CIImage from CVPixelBuffer";
        return false;
      }

      if (normalizedCrop.width > 0.0f && normalizedCrop.height > 0.0f) {
        const CGRect sourceExtent = sourceImage.extent;
        float cropX = normalizedCrop.x;
        if (frame.mirrored) {
          cropX = 1.0f - normalizedCrop.x - normalizedCrop.width;
        }
        const CGRect cropRect = CGRectMake(
            cropX * CGRectGetWidth(sourceExtent),
            normalizedCrop.y * CGRectGetHeight(sourceExtent),
            normalizedCrop.width * CGRectGetWidth(sourceExtent),
            normalizedCrop.height * CGRectGetHeight(sourceExtent));
        sourceImage = [sourceImage imageByCroppingToRect:cropRect];
        if (sourceImage == nil) {
          error = "CIImage crop returned nil";
          return false;
        }
      }

      const CGRect sourceExtent = sourceImage.extent;
      if (CGRectIsEmpty(sourceExtent) || CGRectGetWidth(sourceExtent) <= 0.0 ||
          CGRectGetHeight(sourceExtent) <= 0.0) {
        error = "CIImage has empty extent after crop";
        return false;
      }
      const CGFloat scaleX = static_cast<CGFloat>(targetWidth) / CGRectGetWidth(sourceExtent);
      const CGFloat scaleY = static_cast<CGFloat>(targetHeight) / CGRectGetHeight(sourceExtent);
      CIImage *scaledImage = [sourceImage imageByApplyingTransform:CGAffineTransformMakeScale(scaleX, scaleY)];
      if (scaledImage == nil) {
        error = "CIImage scale returned nil";
        return false;
      }
      [impl_->ciContext_ render:scaledImage toCVPixelBuffer:scaledBuffer];
      auto stageEnd = std::chrono::steady_clock::now();
      scalePerf.addSample(elapsedMs(stageStart, stageEnd));

      stageStart = stageEnd;
      NSError *nsError = nil;
      const CGImagePropertyOrientation orientation =
          frame.mirrored ? kCGImagePropertyOrientationUpMirrored
                         : kCGImagePropertyOrientationUp;
      VNImageRequestHandler *handler =
          [[VNImageRequestHandler alloc] initWithCVPixelBuffer:scaledBuffer
                                                    orientation:orientation
                                                        options:@{}];

      const BOOL success = [handler performRequests:@[ impl_->request_ ]
                                              error:&nsError];
      stageEnd = std::chrono::steady_clock::now();
      requestPerf.addSample(elapsedMs(stageStart, stageEnd));
      if (!success) {
        error = nsError != nil ? nsError.localizedDescription.UTF8String
                               : "Vision request failed";
        return false;
      }

      VNPixelBufferObservation *result = impl_->request_.results.firstObject;
      if (result == nil) {
        error = "Vision request returned no segmentation result";
        return false;
      }

      stageStart = stageEnd;
      const bool copied = copyMaskToProbabilityMat(result.pixelBuffer, targetWidth,
                                                    targetHeight, outProb, error);
      stageEnd = std::chrono::steady_clock::now();
      copyPerf.addSample(elapsedMs(stageStart, stageEnd));
      return copied;
    } @catch (NSException *exception) {
      error = std::string("Vision pipeline ObjC exception: ") +
              exception.name.UTF8String + " - " +
              exception.reason.UTF8String;
      return false;
    }
  }
}

#endif
