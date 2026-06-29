#ifdef __APPLE__

#include "avfoundation_camera.hpp"

#include <chrono>
#include <limits>

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>

namespace {

struct FormatChoice {
  AVCaptureDeviceFormat *format = nil;
  AVFrameRateRange *range = nil;
  double fps = 0.0;
  int pixelCount = std::numeric_limits<int>::max();
};

static cv::Mat convertSampleBufferToBgr(CMSampleBufferRef sampleBuffer) {
  CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
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

static FormatChoice chooseBestFormat(AVCaptureDevice *device) {
  FormatChoice best;
  for (AVCaptureDeviceFormat *format in device.formats) {
    CMVideoDimensions dims =
        CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    const int pixelCount = dims.width * dims.height;
    for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
      const double fps = range.maxFrameRate;
      if (best.format == nil || fps > best.fps + 0.01 ||
          (std::abs(fps - best.fps) <= 0.01 && pixelCount < best.pixelCount)) {
        best.format = format;
        best.range = range;
        best.fps = fps;
        best.pixelCount = pixelCount;
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

    cv::Mat frame = convertSampleBufferToBgr(sampleBuffer);
    if (frame.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(frameMutex_);
    if (latestPixelBuffer_ != nullptr) {
      CVPixelBufferRelease(latestPixelBuffer_);
      latestPixelBuffer_ = nullptr;
    }
    latestPixelBuffer_ = CVPixelBufferRetain(imageBuffer);
    latestFrame_ = std::move(frame);
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
  cv::Mat latestFrame_;
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

bool AvFoundationCamera::open(std::string &error) {
  close();

  @autoreleasepool {
    NSArray<AVCaptureDevice *> *devices =
        [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:@[ AVCaptureDeviceTypeBuiltInWideAngleCamera,
                                               AVCaptureDeviceTypeExternal ]
                                 mediaType:AVMediaTypeVideo
                                  position:AVCaptureDevicePositionUnspecified]
            .devices;

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
    if ([session canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
      session.sessionPreset = AVCaptureSessionPreset1280x720;
    } else if ([session canSetSessionPreset:AVCaptureSessionPresetHigh]) {
      session.sessionPreset = AVCaptureSessionPresetHigh;
    }

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
      const FormatChoice best = chooseBestFormat(device);
      if (best.format != nil && best.range != nil) {
        device.activeFormat = best.format;
        // Use the exact min/max frame duration from the chosen range
        // rather than constructing CMTimeMake(1, round(fps)).  Some
        // cameras (e.g. external displays) have non‑integer frame
        // durations and reject approximations.
        device.activeVideoMinFrameDuration = best.range.minFrameDuration;
        device.activeVideoMaxFrameDuration = best.range.maxFrameDuration;
        impl_->fps_ = best.range.maxFrameRate;
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
    impl_->width_ = dims.width;
    impl_->height_ = dims.height;
    if (impl_->fps_ <= 0.0) {
      impl_->fps_ = device.activeVideoMaxFrameDuration.value != 0
                        ? static_cast<double>(device.activeVideoMaxFrameDuration.timescale) /
                              static_cast<double>(device.activeVideoMaxFrameDuration.value)
                        : 0.0;
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
    impl_->latestFrame_.release();
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
  if (impl_->latestFrame_.empty()) {
    return false;
  }

  frame = impl_->latestFrame_.clone();
  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  impl_->deliveredCounter_ = impl_->frameCounter_;
  return true;
}

bool AvFoundationCamera::latestFrame(cv::Mat &frame) const {
  AppleVideoFrame nativeFrame;
  return latestFrame(frame, nativeFrame);
}

bool AvFoundationCamera::latestFrame(cv::Mat &frame,
                                     AppleVideoFrame &nativeFrame) const {
  if (!impl_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(impl_->frameMutex_);
  if (impl_->latestFrame_.empty()) {
    return false;
  }
  frame = impl_->latestFrame_.clone();
  nativeFrame = AppleVideoFrame(impl_->latestPixelBuffer_, false);
  return true;
}

double AvFoundationCamera::width() const { return impl_ ? impl_->width_ : 0.0; }
double AvFoundationCamera::height() const { return impl_ ? impl_->height_ : 0.0; }
double AvFoundationCamera::fps() const { return impl_ ? impl_->fps_ : 0.0; }
const char *AvFoundationCamera::backendName() const { return "AVFoundation"; }

#endif
