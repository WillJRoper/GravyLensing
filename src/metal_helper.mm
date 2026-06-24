/**
 * @file metal_helper.mm
 *
 * Objective-C++ Metal helper — compiles compute shaders from source and
 * dispatches them on the default GPU device.
 */
#include "metal_helper.h"

#include <vector>

#import <Metal/Metal.h>

namespace {
id<MTLDevice> gDevice = nil;
id<MTLCommandQueue> gQueue = nil;
} // namespace

namespace metal {

struct Pipeline {
  id<MTLComputePipelineState> state;
};

struct Buffers {
  NSMutableArray<id<MTLBuffer>> *bufs;
  std::vector<unsigned long> lengths;
};

bool init() {
  gDevice = MTLCreateSystemDefaultDevice();
  if (!gDevice)
    return false;
  gQueue = [gDevice newCommandQueue];
  return gQueue != nil;
}

Pipeline *createPipeline(const char *name, const char *source) {
  NSError *err = nil;
  NSString *src = [NSString stringWithUTF8String:source];
  id<MTLLibrary> lib = [gDevice newLibraryWithSource:src options:nil error:&err];
  if (!lib) {
    if (err)
      NSLog(@"Metal library compile error: %@", err);
    return nullptr;
  }

  id<MTLFunction> fn =
      [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) {
    NSLog(@"Metal function not found: %s", name);
    return nullptr;
  }

  id<MTLComputePipelineState> state =
      [gDevice newComputePipelineStateWithFunction:fn error:&err];
  if (!state) {
    if (err)
      NSLog(@"Metal pipeline error: %@", err);
    return nullptr;
  }

  auto *p = new Pipeline;
  p->state = state;
  [p->state retain];  // survive without ARC
  return p;
}

void dispatch(Pipeline *p, int gridWidth, const void **buffers,
              const unsigned long *lengths, int bufferCount) {
  if (!p || !gQueue)
    return;

  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:p->state];

  NSMutableArray<id<MTLBuffer>> *mtlBufs =
      [NSMutableArray arrayWithCapacity:bufferCount];

  for (int i = 0; i < bufferCount; ++i) {
    id<MTLBuffer> buf =
        [gDevice newBufferWithBytes:const_cast<void *>(buffers[i])
                             length:lengths[i]
                            options:MTLResourceStorageModeShared];
    [mtlBufs addObject:buf];
    [enc setBuffer:buf offset:0 atIndex:i];
  }

  NSUInteger tgW = p->state.maxTotalThreadsPerThreadgroup;
  if (tgW > 256)
    tgW = 256;
  MTLSize tgSize = MTLSizeMake(tgW, 1, 1);
  MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(gridWidth), 1, 1);
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  for (int i = 0; i < bufferCount; ++i) {
    memcpy(const_cast<void *>(buffers[i]), [mtlBufs[i] contents], lengths[i]);
  }
}

// ── Pre-allocated path ──────────────────────────────────────────────

Buffers *createBuffers(const unsigned long *lengths, int bufferCount) {
  if (!gDevice)
    return nullptr;

  auto *b = new Buffers;
  b->lengths.assign(lengths, lengths + bufferCount);
  b->bufs = [NSMutableArray arrayWithCapacity:bufferCount];
  [b->bufs retain];  // survive without ARC

  for (int i = 0; i < bufferCount; ++i) {
    id<MTLBuffer> buf =
        [gDevice newBufferWithLength:lengths[i]
                             options:MTLResourceStorageModeShared];
    if (!buf) {
      [b->bufs release];
      delete b;
      return nullptr;
    }
    [b->bufs addObject:buf];
  }
  return b;
}

void dispatchWithBuffers(Pipeline *p, int gridWidth, Buffers *b,
                         const void **dataPtrs, int bufferCount) {
  if (!p || !b || !gQueue)
    return;

  // Upload CPU data into the pre-allocated GPU buffers & bind them.
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:p->state];

  for (int i = 0; i < bufferCount; ++i) {
    id<MTLBuffer> buf = b->bufs[i];
    memcpy([buf contents], dataPtrs[i], b->lengths[i]);
    [enc setBuffer:buf offset:0 atIndex:i];
  }

  NSUInteger tgW = p->state.maxTotalThreadsPerThreadgroup;
  if (tgW > 256)
    tgW = 256;
  MTLSize tgSize = MTLSizeMake(tgW, 1, 1);
  MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(gridWidth), 1, 1);
  [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  // Download output data back to the caller's CPU pointers.
  for (int i = 0; i < bufferCount; ++i) {
    memcpy(const_cast<void *>(dataPtrs[i]), [b->bufs[i] contents],
           b->lengths[i]);
  }
}

void releasePipeline(Pipeline *p) {
  if (p) {
    [p->state release];
    delete p;
  }
}

void releaseBuffers(Buffers *b) {
  if (b) {
    [b->bufs release];
    delete b;
  }
}

} // namespace metal
