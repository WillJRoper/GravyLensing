#ifdef __APPLE__

#include "avfoundation_camera.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>

namespace {

struct FormatChoice {
  AVCaptureDeviceFormat *format = nil;
  AVFrameRateRange *range = nil;
  double fps = 0.0;
  double fpsDistance = std::numeric_limits<double>::max();
  int pixelCount = std::numeric_limits<int>::max();
  CMTime frameDuration = kCMTimeInvalid;
};

static NSArray<AVCaptureDevice *> *discoverVideoDevices() {
  return [AVCaptureDeviceDiscoverySession
             discoverySessionWithDeviceTypes:@[ AVCaptureDeviceTypeBuiltInWideAngleCamera,
                                                AVCaptureDeviceTypeExternal ]
                                  mediaType:AVMediaTypeVideo
                                   position:AVCaptureDevicePositionUnspecified]
      .devices;
}

static cv::Mat convertPixelBufferToBgr(CVPixelBufferRef imageBuffer) {
  if (imageBuffer == nullptr) {
    return cv::Mat();
  }

  CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
  const int width = static_cast<int>(CVPixelBufferGetWidth(imageBuffer));
  const int height = static_cast<int>(CVPixelBufferGetHeight(imageBuffer));
  void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
  const size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);

  cv::Mat bgra(height, width, CV_8UC4, baseAddress, bytesPerRow);
  cv::Mat bgr;
  cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

  CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
  return bgr;
}

static FormatChoice chooseBestFormat(AVCaptureDevice *device,
                                     int desiredFps, int desiredWidth,
                                     int desiredHeight) {
  FormatChoice best;
  const int targetWidth = desiredWidth > 0 ? desiredWidth : 1280;
  const int targetHeight = desiredHeight > 0 ? desiredHeight : 720;
  for (AVCaptureDeviceFormat *format in device.formats) {
    CMVideoDimensions dims =
        CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    const int pixelCount = dims.width * dims.height;
    for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
      const double targetFps = static_cast<double>(desiredFps);
      const double fps = std::clamp(targetFps, range.minFrameRate,
                                    range.maxFrameRate);
      const double fpsDistance = std::abs(fps - targetFps);
      const int sizeDistance = std::abs(dims.width - targetWidth) +
                               std::abs(dims.height - targetHeight);
      const CMVideoDimensions bestDims =
          best.format != nil
              ? CMVideoFormatDescriptionGetDimensions(best.format.formatDescription)
              : CMVideoDimensions{0, 0};
      const int bestSizeDistance = std::abs(bestDims.width - targetWidth) +
                                   std::abs(bestDims.height - targetHeight);
      if (best.format == nil || fpsDistance < best.fpsDistance - 0.01 ||
          (std::abs(fpsDistance - best.fpsDistance) <= 0.01 &&
           sizeDistance < bestSizeDistance)) {
        best.format = format;
        best.range = range;
        best.fps = fps;
        best.fpsDistance = fpsDistance;
        best.pixelCount = pixelCount;
        if (std::abs(fps - range.maxFrameRate) <= 0.01)
          best.frameDuration = range.minFrameDuration;
        else if (std::abs(fps - range.minFrameRate) <= 0.01)
          best.frameDuration = range.maxFrameDuration;
        else
          best.frameDuration = CMTimeMakeWithSeconds(1.0 / fps, 1000000000);
      }
    }
  }
  return best;
}

} // namespace

class AvFoundationCamera::Impl {
public:
  explicit Impl(int deviceIndex) : deviceIndex_(deviceIndex) {}

  void onFrame(CMSampleBufferRef sampleBuffer) {
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (imageBuffer == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(frameMutex_);
    width_ = static_cast<int>(CVPixelBufferGetWidth(imageBuffer));
    height_ = static_cast<int>(CVPixelBufferGetHeight(imageBuffer));
    if (latestPixelBuffer_ != nullptr) {
      CVPixelBufferRelease(latestPixelBuffer_);
      latestPixelBuffer_ = nullptr;
    }
    latestPixelBuffer_ = CVPixelBufferRetain(imageBuffer);
    ++frameCounter_;
    frameCv_.notify_all();
  }

  int deviceIndex_ = 0;
  AVCaptureSession *session_ = nil;
  AVCaptureDeviceInput *input_ = nil;
  AVCaptureVideoDataOutput *output_ = nil;
  dispatch_queue_t outputQueue_ = nullptr;
  NSObject *delegate_ = nil;
  double width_ = 0.0;
  double height_ = 0.0;
  double fps_ = 0.0;
  mutable std::mutex frameMutex_;
  mutable std::condition_variable frameCv_;
  CVPixelBufferRef latestPixelBuffer_ = nullptr;
  uint64_t frameCounter_ = 0;
  uint64_t deliveredCounter_ = 0;
};

@interface GravyCaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(nonatomic, assign) void *owner;
@end

@implementation GravyCaptureDelegate
- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
  (void)output;
  (void)connection;
  if (self.owner != nullptr) {
    static_cast<AvFoundationCamera::Impl *>(self.owner)->onFrame(sampleBuffer);
  }
}
@end

AvFoundationCamera::AvFoundationCamera(int deviceIndex)
    : impl_(std::make_unique<Impl>(deviceIndex)) {}

AvFoundationCamera::~AvFoundationCamera() { close(); }

std::vector<std::string> AvFoundationCamera::availableDeviceNames() {
  std::vector<std::string> names;
  @autoreleasepool {
    for (AVCaptureDevice *device in discoverVideoDevices()) {
      names.emplace_back(device.localizedName.UTF8String);
    }
  }
  return names;
}

bool AvFoundationCamera::open(std::string &error, int desiredFps,
                              int desiredWidth, int desiredHeight) {
  close();

  @autoreleasepool {
    NSArray<AVCaptureDevice *> *devices = discoverVideoDevices();

    if (impl_->deviceIndex_ < 0 || impl_->deviceIndex_ >= static_cast<int>(devices.count)) {
      error = "Camera device index out of range";
      return false;
    }

    AVCaptureDevice *device = devices[impl_->deviceIndex_];
    NSError *nsError = nil;
    AVCaptureDeviceInput *input =
        [AVCaptureDeviceInput deviceInputWithDevice:device error:&nsError];
    if (input == nil) {
      error = nsError != nil ? nsError.localizedDescription.UTF8String
                             : "Failed to create AVCaptureDeviceInput";
      return false;
    }

    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    if (![session canAddInput:input]) {
      error = "Failed to add AVCapture input";
      return false;
    }
    [session addInput:input];

    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    output.alwaysDiscardsLateVideoFrames = YES;
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey:
                                 @(kCVPixelFormatType_32BGRA)};
    if (![session canAddOutput:output]) {
      error = "Failed to add AVCapture output";
      return false;
    }
    [session addOutput:output];

    AVCaptureConnection *connection = [output connectionWithMediaType:AVMediaTypeVideo];
    if ([connection isVideoMirroringSupported]) {
      connection.videoMirrored = NO;
    }

    if ([device lockForConfiguration:&nsError]) {
      if ([device isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure]) {
        device.exposureMode = AVCaptureExposureModeContinuousAutoExposure;
      }
       const FormatChoice best = chooseBestFormat(
           device, desiredFps, desiredWidth, desiredHeight);
      if (best.format != nil && best.range != nil) {
        device.activeFormat = best.format;
        // Use the exact min/max frame duration from the chosen range
        // rather than constructing CMTimeMake(1, round(fps)).  Some
        // cameras (e.g. external displays) have non‑integer frame
        // durations and reject approximations.
        device.activeVideoMinFrameDuration = best.frameDuration;
        device.activeVideoMaxFrameDuration = best.frameDuration;
        impl_->fps_ = 1.0 / CMTimeGetSeconds(best.frameDuration);
      }
      [device unlockForConfiguration];
    }

    GravyCaptureDelegate *delegate = [[GravyCaptureDelegate alloc] init];
    delegate.owner = impl_.get();
    dispatch_queue_t outputQueue =
        dispatch_queue_create("com.gravylensing.capture", DISPATCH_QUEUE_SERIAL);
    [output setSampleBufferDelegate:delegate queue:outputQueue];

    [session startRunning];

    impl_->session_ = session;
    impl_->input_ = input;
    impl_->output_ = output;
    impl_->outputQueue_ = outputQueue;
    impl_->delegate_ = delegate;

    AVCaptureDeviceFormat *activeFormat = device.activeFormat;
    CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(activeFormat.formatDescription);
    {
      std::lock_guard<std::mutex> lock(impl_->frameMutex_);
      impl_->width_ = dims.width;
      impl_->height_ = dims.height;
      if (impl_->fps_ <= 0.0) {
        impl_->fps_ = device.activeVideoMaxFrameDuration.value != 0
                          ? static_cast<double>(device.activeVideoMaxFrameDuration.timescale) /
                                static_cast<double>(device.activeVideoMaxFrameDuration.value)
                          : 0.0;
      }
    }
  }

  cv::Mat warmup;
  return waitForFrame(warmup, 2000, nullptr);
}

void AvFoundationCamera::close() {
  if (!impl_) {
    return;
  }

  @autoreleasepool {
    if (impl_->output_ != nil) {
      [impl_->output_ setSampleBufferDelegate:nil queue:nullptr];
    }
    if (impl_->session_ != nil) {
      [impl_->session_ stopRunning];
    }
  }

  {
    std::lock_guard<std::mutex> lock(impl_->frameMutex_);
    if (impl_->latestPixelBuffer_ != nullptr) {
      CVPixelBufferRelease(impl_->latestPixelBuffer_);
      impl_->latestPixelBuffer_ = nullptr;
    }
    impl_->frameCounter_ = 0;
    impl_->deliveredCounter_ = 0;
  }

  impl_->session_ = nil;
  impl_->input_ = nil;
  impl_->output_ = nil;
  impl_->delegate_ = nil;
  impl_->outputQueue_ = nullptr;
  impl_->width_ = 0.0;
  impl_->height_ = 0.0;
  impl_->fps_ = 0.0;
}

bool AvFoundationCamera::waitForFrame(cv::Mat &frame, int timeoutMs,
                                      const std::atomic<bool> *stopRequested) {
  AppleVideoFrame nativeFrame;
  return waitForFrame(frame, nativeFrame, timeoutMs, stopRequested);
}

bool AvFoundationCamera::waitForNativeFrame(AppleVideoFrame &nativeFrame,
                                            int timeoutMs,
                                            const std::atomic<bool> *stopRequested) {
  if (!impl_) {
    return false;
  }

  std::unique_lock<std::mutex> lock(impl_->frameMutex_);
  const auto predicate = [&]() {
    return impl_->frameCounter_ != impl_->deliveredCounter_ ||
           (stopRequested != nullptr && stopRequested->load());
  };

  if (timeoutMs <= 0) {
    impl_->frameCv_.wait(lock, predicate);
  } else if (!impl_->frameCv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                                       predicate)) {
    return false;
  }

  if (stopRequested != nullptr && stopRequested->load()) {
    return false;
  }
  if (impl_->latestPixelBuffer_ == nullptr) {
    return false;
  }

  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  impl_->deliveredCounter_ = impl_->frameCounter_;
  return true;
}

bool AvFoundationCamera::waitForFrame(cv::Mat &frame, AppleVideoFrame &nativeFrame,
                                      int timeoutMs,
                                      const std::atomic<bool> *stopRequested) {
  if (!impl_) {
    return false;
  }

  std::unique_lock<std::mutex> lock(impl_->frameMutex_);
  const auto predicate = [&]() {
    return impl_->frameCounter_ != impl_->deliveredCounter_ ||
           (stopRequested != nullptr && stopRequested->load());
  };

  if (timeoutMs <= 0) {
    impl_->frameCv_.wait(lock, predicate);
  } else if (!impl_->frameCv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                                       predicate)) {
    return false;
  }

  if (stopRequested != nullptr && stopRequested->load()) {
    return false;
  }
  if (impl_->latestPixelBuffer_ == nullptr) {
    return false;
  }

  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  frame = convertPixelBufferToBgr(nativeFrame.pixelBuffer);
  if (frame.empty()) {
    return false;
  }
  impl_->deliveredCounter_ = impl_->frameCounter_;
  return true;
}

bool AvFoundationCamera::latestFrame(cv::Mat &frame) const {
  AppleVideoFrame nativeFrame;
  return latestFrame(frame, nativeFrame);
}

bool AvFoundationCamera::latestNativeFrame(AppleVideoFrame &nativeFrame) const {
  if (!impl_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  if (impl_->latestPixelBuffer_ == nullptr) {
    return false;
  }
  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  return true;
}

bool AvFoundationCamera::latestFrame(cv::Mat &frame,
                                     AppleVideoFrame &nativeFrame) const {
  if (!impl_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  if (impl_->latestPixelBuffer_ == nullptr) {
    return false;
  }
  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  frame = convertPixelBufferToBgr(nativeFrame.pixelBuffer);
  if (frame.empty()) {
    return false;
  }
  return true;
}

double AvFoundationCamera::width() const {
  if (!impl_)
    return 0.0;
  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  return impl_->width_;
}
double AvFoundationCamera::height() const {
  if (!impl_)
    return 0.0;
  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  return impl_->height_;
}
double AvFoundationCamera::fps() const {
  if (!impl_)
    return 0.0;
  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  return impl_->fps_;
}
const char *AvFoundationCamera::backendName() const { return "AVFoundation"; }

#endif
