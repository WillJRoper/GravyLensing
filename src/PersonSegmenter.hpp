#pragma once
#import <CoreVideo/CoreVideo.h>
#include <functional>

/// A thin ObjC++ wrapper around Vision.person segmentation.
class PersonSegmenter {
public:
  using Callback = std::function<void(CVPixelBufferRef maskBuffer)>;

  PersonSegmenter();
  ~PersonSegmenter();

  /// Asynchronously segment the BGRA frameBuffer. When done, calls
  /// cb(maskBuffer). You must CFRelease maskBuffer when you're done.
  void segmentFrame(CVPixelBufferRef frameBuffer, Callback cb);

private:
  void *_handler; // VNSequenceRequestHandler*
  void *_request; // VNGeneratePersonSegmentationRequest*
};
