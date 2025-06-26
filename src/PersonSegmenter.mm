#import "PersonSegmenter.hpp"
#import <Vision/Vision.h>

@implementation PersonSegmenter {
  VNSequenceRequestHandler *handler;
  VNGeneratePersonSegmentationRequest *request;
}

- (instancetype)init {
  if ((self = [super init])) {
    handler = [[VNSequenceRequestHandler alloc] init];
    request = [[VNGeneratePersonSegmentationRequest alloc] init];
    request.qualityLevel = VNGeneratePersonSegmentationRequestQualityLevelFast;
    // single channel 0–255 mask:
    request.outputPixelFormat = kCVPixelFormatType_OneComponent8;
  }
  return self;
}

- (void)segmentFrame:(CVPixelBufferRef)frameBuffer
            callback:(void (^)(CVPixelBufferRef))cb {
  CVBufferRetain(frameBuffer);
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
    NSError *err = nil;
    [handler performRequests:@[ request ]
             onCVPixelBuffer:frameBuffer
                       error:&err];
    CVPixelBufferRef mask = nil;
    if (!err) {
      VNPixelBufferObservation *obs =
          (VNPixelBufferObservation *)request.results.firstObject;
      mask = obs.pixelBuffer;
      CVBufferRetain(mask);
    }
    CVBufferRelease(frameBuffer);
    cb(mask);
    if (mask)
      CVBufferRelease(mask);
  });
}

- (void)dealloc {
  // ARC will clean handler & request
}

@end
