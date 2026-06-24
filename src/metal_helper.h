/**
 * @file metal_helper.h
 *
 * Minimal Metal helper — compile compute shaders from source at runtime
 * and dispatch them.  Only used on Apple platforms (USE_MPS).
 *
 * Two dispatch modes:
 *   dispatch()          — one-shot: allocates, uploads, runs, downloads.
 *   createBuffers() + dispatchWithBuffers()
 *                       — pre-allocated: reuse buffers across frames.
 */
#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#else
#include <cstdint>
#endif

namespace metal {

/// Opaque handle to a compiled compute pipeline.
struct Pipeline;

/// Opaque handle to a set of pre-allocated Metal buffers.
struct Buffers;

/// Initialise the Metal device and default command queue.  Call once.
bool init();

/// Compile a compute kernel from source and create a pipeline.
Pipeline *createPipeline(const char *name, const char *source);

/// One-shot dispatch: allocate buffers, upload data, run kernel, download.
void dispatch(Pipeline *p, int gridWidth, const void **buffers,
              const unsigned long *lengths, int bufferCount);

/// Pre-allocate GPU buffers matching the given sizes.  Call once (or when
/// geometry changes).  The returned handle is reused by dispatchWithBuffers().
Buffers *createBuffers(const unsigned long *lengths, int bufferCount);

/// Dispatch using pre-allocated buffers.  dataPtrs point to the CPU memory
/// to upload (for inputs) and receive results (for outputs).
void dispatchWithBuffers(Pipeline *p, int gridWidth, Buffers *bufs,
                         const void **dataPtrs, int bufferCount);

/// Destroy a pipeline handle.
void releasePipeline(Pipeline *p);

/// Destroy a pre-allocated buffer set.
void releaseBuffers(Buffers *b);

} // namespace metal
