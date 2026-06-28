#pragma once

#ifdef __APPLE__

#include <CoreVideo/CoreVideo.h>

struct AppleVideoFrame {
  CVPixelBufferRef pixelBuffer = nullptr;
  bool mirrored = false;

  AppleVideoFrame() = default;

  AppleVideoFrame(CVPixelBufferRef buffer, bool isMirrored)
      : pixelBuffer(buffer), mirrored(isMirrored) {
    if (pixelBuffer != nullptr) {
      CFRetain(pixelBuffer);
    }
  }

  AppleVideoFrame(const AppleVideoFrame &other)
      : pixelBuffer(other.pixelBuffer), mirrored(other.mirrored) {
    if (pixelBuffer != nullptr) {
      CFRetain(pixelBuffer);
    }
  }

  AppleVideoFrame(AppleVideoFrame &&other) noexcept
      : pixelBuffer(other.pixelBuffer), mirrored(other.mirrored) {
    other.pixelBuffer = nullptr;
    other.mirrored = false;
  }

  AppleVideoFrame &operator=(const AppleVideoFrame &other) {
    if (this == &other) {
      return *this;
    }
    if (pixelBuffer != nullptr) {
      CFRelease(pixelBuffer);
    }
    pixelBuffer = other.pixelBuffer;
    mirrored = other.mirrored;
    if (pixelBuffer != nullptr) {
      CFRetain(pixelBuffer);
    }
    return *this;
  }

  AppleVideoFrame &operator=(AppleVideoFrame &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (pixelBuffer != nullptr) {
      CFRelease(pixelBuffer);
    }
    pixelBuffer = other.pixelBuffer;
    mirrored = other.mirrored;
    other.pixelBuffer = nullptr;
    other.mirrored = false;
    return *this;
  }

  ~AppleVideoFrame() {
    if (pixelBuffer != nullptr) {
      CFRelease(pixelBuffer);
    }
  }

  bool isValid() const { return pixelBuffer != nullptr; }
};

#endif
