
#include "PixelBufferUtils.hpp"
#include "VisionSegmentationWorker.hpp"

VisionSegmentationWorker::VisionSegmentationWorker(QObject *parent)
    : QObject(parent), _seg() {}

void VisionSegmentationWorker::onFrame(const cv::Mat &frame) {
  // Convert to BGRA if needed:
  cv::Mat bgra;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, bgra, cv::COLOR_BGR2BGRA);
  } else {
    bgra = frame;
  }

  CVPixelBufferRef buf = matToPixelBuffer(bgra);
  _seg.segmentFrame(buf, [this](CVPixelBufferRef maskBuf) {
    // maskBuf is 0–255 one‐channel
    cv::Mat mask = pixelBufferToMat(maskBuf);
    CVBufferRelease(maskBuf);
    emit maskReady(mask);
  });
  CVBufferRelease(buf);
}
