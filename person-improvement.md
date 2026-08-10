# Person Detection Improvement Plan

## Goal

Produce a high-quality real-time person mask on macOS, including multiple people, and feed that mask into the existing lensing stage with minimal latency and stable edges.

## Current State

The current person-mask pipeline lives in `src/segmentation_worker.cpp` and uses TorchScript segmentation on top of LibTorch, typically through the MPS backend on macOS.

The current stages are:

1. Resize frame to model resolution
2. Convert BGR to RGB
3. Pack pixels into a tensor
4. Upload tensor to device
5. Normalize input
6. Run inference
7. Extract person probability map
8. Apply temporal smoothing
9. Threshold to binary mask
10. Morphological cleanup and tiny-blob removal
11. Upscale mask to the lensing resolution

This is functional, but it is not yet at the standard expected for conferencing-grade person segmentation on macOS.

## Guiding Principles

1. Use macOS-native inference for the main path.
2. Preserve support for multiple people in frame.
3. Keep improvements measurable with profiling and visual testing.
4. Improve mask quality before spending time on small micro-optimizations.
5. Avoid changing the lensing pipeline until mask quality clearly improves.

## Cross-Mode Relevance

This document is focused on person mode, but several parts of the plan also benefit color mode.

Likely person-mode-only work:

1. Vision-based segmentation backend
2. Vision request tuning and native Apple segmentation plumbing
3. Person-specific ROI acceleration for the Vision path

Likely useful for both person mode and color mode:

1. Better temporal stabilization concepts
2. Better mask-to-lensing integration
3. Runtime quality and performance modes
4. Profiling discipline and stage-level performance measurement
5. Any improvements to how soft or binary masks are converted into lensing mass

Possibly useful for color mode with adaptation:

1. Boundary refinement ideas, if color mode begins retaining a soft confidence mask instead of only a hard binary mask
2. ROI acceleration ideas, if color tracking later uses a stable confidence field rather than only blob extraction
3. Image-guided edge cleanup, if color mode starts suffering from weak bridges or unstable mask expansion

## Profiling

Profiling has been added to `src/segmentation_worker.cpp` with these buckets:

- `person-mask`
- `person-mask-resize`
- `person-mask-color`
- `person-mask-pack`
- `person-mask-upload`
- `person-mask-normalize`
- `person-mask-infer`
- `person-mask-prob`
- `person-mask-cleanup`
- `person-mask-upscale`

Enable profiling with:

```bash
cmake -B build -DENABLE_PROFILING=ON
cmake --build build
```

## Stage 1 Findings

Stage 1 has now been run on the current Torch/MPS pipeline with:

```bash
./gravy_lens --nthreads 12 --softening 50 --strength 4 --lowerRes 0.5 --flip --distortInside --debugGrid
```

Observed behavior:

1. Person detection was visibly blobby.
2. Extremities were inconsistent, especially hands and other thin or fast-moving regions.
3. Overall mask quality was not yet good enough for a polished final effect.

Measured steady-state performance was still strong:

1. Capture stayed near 30 FPS.
2. Person-mask total time stabilized around `10.5-13.7 ms`.
3. Lensing total time stabilized around `18.2-19.1 ms`.
4. The app sustained roughly 30 FPS overall once warmed up.

Important profiling takeaways:

1. Inference is a major cost, but not the only one.
2. `person-mask-prob` is also a significant and sometimes highly variable cost.
3. Preprocessing costs are relatively small compared with inference and probability extraction.
4. The current pipeline is already fast enough to support real-time behavior at 30 FPS on the tested hardware.

Main conclusion from Stage 1:

The primary problem is now mask quality, not raw throughput. Performance still matters, but the next meaningful work should focus on replacing the current segmentation path with a higher-quality macOS-native approach and then improving mask representation and refinement.

## End-To-End Plan

### Phase 1: Baseline The Current Pipeline

Collect profiling and behavior data for representative scenes:

1. One person centered
2. Multiple people in frame
3. Fast arm and head motion
4. Partial occlusion
5. Hair and detail-heavy scenes
6. Cluttered backgrounds

Record:

1. Total mask latency
2. Per-stage latency
3. Effective mask FPS
4. Frame-to-frame latency variance
5. Visual notes on flicker, lag, and edge quality

Deliverable:

- A baseline performance and quality table for the current Torch/MPS path

Status:

- Completed

Outcome:

- Real-time throughput is already acceptable on the tested Mac.
- Current mask quality is the limiting factor.
- The existing Torch/MPS path is a useful baseline and fallback, but not the preferred long-term person-mask implementation.

### Phase 2: Add A macOS-Native Segmentation Path

Implement a new macOS-specific person segmentation worker using Apple's Vision framework.

Plan:

1. Add a new worker, likely `vision_segmentation_worker.hpp` and `vision_segmentation_worker.mm`
2. Use `VNGeneratePersonSegmentationRequest`
3. Feed it `CVPixelBuffer` frames directly from AVFoundation
4. Avoid the current resize, color conversion, tensor packing, and device upload overhead
5. Keep support for multiple people by using the full person segmentation output
6. Keep the current Torch-based worker as a fallback path if needed

Why this is the primary change:

1. It removes a large amount of pipeline overhead
2. It uses Apple's optimized on-device path
3. It is the most likely route to both better quality and better real-time performance on Mac hardware

Deliverable:

- Vision-backed end-to-end person mask generation on macOS

Priority after Stage 1:

- Highest

Reason:

- The current path is fast enough, but the quality is not good enough.
- The biggest remaining opportunity is to improve segmentation quality without sacrificing real-time performance.

Current findings:

1. A first Vision integration was implemented.
2. That first pass was far too slow, with `person-mask-vision` around `102-152 ms` and total person-mask time around `103-153 ms`.
3. This reduced end-to-end throughput to roughly `6.5-9.7 FPS`, which is not acceptable.

Cause of the regression:

1. Vision was being run on full-resolution camera frames.
2. The result was only reduced to the configured segmentation resolution after inference.
3. That erased the performance advantage of using a smaller working resolution.

Correction now applied:

1. The Vision backend has been changed to resize input to the configured segmentation resolution before inference.
2. The default Vision quality level has been reduced from `Accurate` to `Balanced` for the baseline real-time path.
3. The Vision backend now consumes native `CVPixelBuffer` camera frames instead of round-tripping person mode through `cv::Mat`.
4. The scaled working `CVPixelBuffer` is now reused across frames.

Latest profiling result:

1. `person-mask-vision` is now roughly `32.7-34.0 ms` in steady state.
2. `person-mask-vision-scale` is only about `1.6-2.7 ms`.
3. `person-mask-vision-copy` is only about `0.14-0.42 ms`.
4. `person-mask-vision-request` is the dominant cost at about `29.8-31.4 ms`.
5. Overall app throughput remains near `30 FPS`, but the person segmentation stage itself has little spare headroom.

Conclusion:

1. The quality improvement from Vision is real.
2. Most avoidable integration overhead has already been removed.
3. Remaining performance cost is dominated by Vision inference itself, not by conversion or scaling overhead.
4. Additional large performance gains inside Phase 2 are unlikely without changing working resolution, request quality level, or segmentation strategy.
5. The correct next step is to continue with mask-quality phases rather than spending excessive time chasing tiny integration wins.

Phase 2 exit criteria:

1. Restore real-time performance close to the Stage 1 baseline.
2. Improve or at least preserve mask quality relative to the Torch/MPS path.
3. If quality improves but performance remains unacceptable, Phase 2 is not complete and must continue before moving on.

### Phase 3: Keep A Soft Mask Longer

The current pipeline converts the person probability map into a binary mask too early.

We should change the internal flow to:

1. Keep a float probability or alpha mask internally
2. Apply temporal stabilization in probability space
3. Refine boundaries before hard thresholding
4. Only create a binary mask when the lensing stage actually requires it

Implementation notes:

1. Add explicit `CV_32F` mask handling for the primary path
2. Maintain both a soft mask and a binary mask where needed
3. Avoid treating thresholding as the main refinement stage

Deliverable:

- A soft-mask-based internal pipeline that preserves more detail and supports cleaner refinement

Priority after Stage 1:

- High

Reason:

- The current binary mask path contributes to blobby silhouettes and unstable extremities.

Current implementation status:

1. The segmentation pipeline now retains and refines a soft probability mask internally before converting to a binary mask.
2. Hysteresis-style thresholding has replaced the old single `0.5` threshold.
3. An initial iterative region-growth cleanup approach improved quality but introduced severe cleanup-time spikes in some scenes.
4. That cleanup step has been replaced with a linear-time connected-components hysteresis pass that keeps weak-foreground regions only when they are supported by strong foreground.

Current conclusion:

1. Soft-mask retention is still the right direction.
2. Cleanup logic must remain bounded and predictable in runtime cost.
3. Quality improvements are only acceptable if they do not introduce major frame-time spikes.

### Phase 4: Improve Boundary Quality

Hair, hands, and shoulders are the most important visible boundary areas.

Add refinement in this order:

1. Probability hysteresis with clear foreground and background thresholds
2. Edge-aware refinement using the input frame
3. Limited morphological cleanup after refinement
4. Optional feathered edge output if it improves final lensing quality

Candidate refinement tools:

1. Guided filter
2. Joint bilateral filter
3. Boundary-band-only smoothing and cleanup

Avoid:

1. Large morphological kernels as the main cleanup tool
2. Heavy close/open operations that erase fingers, hair, or clothing detail

Deliverable:

- Cleaner mask edges with better preservation of fine structure

Priority after Stage 1:

- High

Reason:

- Boundary quality is the most obvious visible weakness in the current output.

Current implementation status:

1. The mask refinement step is now image-aware.
2. Weak-confidence uncertain regions are pruned using camera-frame edge information before final growth and cleanup.
3. This is intended to reduce unstable weak bridges that incorrectly connect nearby body parts, such as flickering arm-to-torso or head-to-shoulder links.

Current conclusion:

1. Image-guided refinement is necessary because probability-space cleanup alone is too eager to fill weak gaps.
2. The refinement must stay lightweight enough to preserve real-time performance.

### Phase 5: Improve Temporal Stability Without Adding Lag

The current temporal EMA is simple but creates a direct tradeoff between flicker and lag.

Replace it with adaptive temporal behavior:

1. Stronger smoothing in stable interior regions
2. Weaker smoothing at moving boundaries
3. Lower history influence when the scene changes rapidly

Practical first implementation:

1. Identify a boundary band around the current mask
2. Use lower temporal smoothing inside that band
3. Use higher temporal smoothing in confident interior regions

Possible later extension:

1. Motion-aware blending
2. Optical-flow-assisted mask warping

Deliverable:

- More stable masks with less visible trailing and less flicker

Current implementation status:

1. Global fixed temporal blending has been replaced with adaptive per-pixel temporal smoothing.
2. The blend factor now increases in uncertain boundary regions and where the probability map changes rapidly.
3. Stable interior regions retain stronger history, while moving or uncertain edges are allowed to update faster.
4. The active segmentation pipeline now actually applies that adaptive temporal blend after the short median stage, instead of only documenting it.

Current conclusion:

1. Temporal behavior must not be uniform across the whole mask.
2. Boundary and motion awareness are necessary to reduce flicker without making the whole silhouette sticky.

Priority after Stage 1:

- High

Reason:

- Extremity inconsistency strongly suggests the current temporal behavior is not preserving moving boundaries well enough.

### Phase 6: Add ROI-Based Acceleration

Once the core mask quality is strong, reduce average compute cost.

Plan:

1. Run full-frame segmentation at a lower cadence
2. Build a padded union ROI around all active person regions
3. Run higher-rate segmentation within that ROI
4. Periodically fall back to full-frame reacquisition to catch new entrants and exits

This preserves multiple-person support while reducing wasted work on empty background.

Deliverable:

- Lower average segmentation cost in typical scenes without losing robustness

Current implementation status:

1. The Vision path now supports cropped ROI requests.
2. ROI is now derived from a soft probability field rather than only the final binary mask.
3. The pipeline periodically reacquires a full-frame segmentation to avoid lock-in and recover from large motion or new entrants.
4. ROI request resolution scales with ROI size rather than always using the full working resolution.
5. ROI activation is now gated by confidence, area fraction, and overlap with the previous ROI to reduce visible strobing.
6. ROI updates now carry inertia by unioning with the previous stable ROI and force a full-frame reacquisition when confidence collapses.

Current conclusion:

1. Future 60 FPS viability depends on reducing Vision request work, not only refining surrounding code.
2. ROI-based request reduction is the most meaningful remaining performance lever in the current architecture.
3. The first ROI implementation introduced severe visible strobing, but the revised design now uses stronger hysteresis and confidence gating to reduce that risk.
4. ROI acceleration should still be treated as quality-sensitive and may need scene testing before being considered the default in every preset.

### Phase 7: Clean Up The Camera-To-Mask Data Path

On macOS, keep frames in `CVPixelBuffer` form as long as possible.

Tasks:

1. Avoid converting camera frames to `cv::Mat` before segmentation on the native path
2. Convert to OpenCV types only where the rest of the app still requires them
3. Keep a minimal conversion path for debug display if necessary

Deliverable:

- Lower copy overhead and a cleaner frame ownership model

### Phase 8: Refine How The Mask Drives Lensing

After the mask pipeline improves, reassess how lensing should consume it.

Evaluate whether the lensing input should be:

1. A binary mask
2. A soft alpha mask
3. A smoothed or distance-like mass map derived from the mask

The current hard binary silhouette may not be the best visual driver for the lensing effect.

Deliverable:

- Better visual integration between the person mask and the lensing output

Current implementation status:

1. The lensing stage now consumes a softened float mass map derived from the refined binary mask rather than the raw hard edge alone.
2. This preserves the existing mask-generation pipeline while reducing harsh mass discontinuities at person boundaries.

Current conclusion:

1. A soft mass map is a low-risk improvement because it does not require a new detector backend or a UI-facing mode split.
2. Boundary softness should stay lightweight so the extra pre-FFT work does not meaningfully erode the current 30 FPS envelope.

### Phase 9: Add Runtime Quality Modes

Expose a small set of user-facing quality modes:

1. `Fast`
2. `Balanced`
3. `High Quality`

These modes should control:

1. Vision request quality level
2. Temporal stabilization strength
3. Boundary refinement strength
4. ROI cadence
5. Optional soft-edge behavior

Deliverable:

- Clear, predictable runtime tradeoffs for different Mac hardware and use cases

Current implementation status:

1. A session-level quality-mode setting now exposes `Fast`, `Balanced`, `High Quality`, and `Custom`.
2. The preset currently drives Vision request quality, working segmentation resolution, temporal stabilization aggressiveness, and lens mass softening strength together.
3. The same preset also feeds the shared runtime resolution scale used by color mode so the user-facing tradeoff stays consistent across mask modes.
4. `Fast` now pushes the working resolution lower and also skips image-guided refinement when the Apple path is running without preview frames.

Current conclusion:

1. A small preset surface is more usable than expecting manual tuning of several coupled controls.
2. `Custom` remains available when the preset buckets are too coarse.

### Phase 10: Verification And Acceptance Criteria

Define success with both performance and quality targets.

Performance targets:

1. Real-time sustained operation on target Mac hardware
2. Stable mask FPS with low latency variance
3. No pipeline backlog in normal use

Quality targets:

1. Supports multiple people at once
2. Stable silhouettes with low flicker
3. Better retention of hair and hands than the current pipeline
4. Fast response to entry, exit, and motion without heavy lag

Test scenes:

1. Single subject
2. Multiple subjects
3. Crossing subjects
4. Seated and standing subjects
5. Fast arm motion
6. Partial occlusion
7. Busy background
8. Backlit and low-contrast clothing

Deliverable:

- A repeatable validation checklist for performance and quality

## Recommended Implementation Order

1. Baseline the current pipeline using the new profiling
2. Implement the Vision-based person segmentation path
3. Convert the internal pipeline to retain soft masks
4. Add boundary refinement
5. Add adaptive temporal stabilization
6. Tune how the refined mask drives lensing
7. Revisit ROI-based acceleration with a more robust design
8. Add runtime quality modes
9. Benchmark, compare, and finalize defaults

## Immediate Next Steps

Based on the current checkpoint, the next work should be:

1. Re-evaluate whether lensing should consume a refined binary mask or a softened mass map.
2. Benchmark the stable Vision + refinement path as the accepted baseline.
3. Revisit ROI acceleration later with a more robust design built around soft-mask confidence and stronger ROI hysteresis.
4. Add runtime quality modes so performance can be traded against quality intentionally.

## Lower-Priority Work

These are not the right first targets:

1. Swapping between the two existing Torch models
2. Spending significant time on micro-optimizing the current Torch preprocessing loops
3. Tuning morphology kernels in isolation

These may still help later, but they are smaller wins than moving to the native macOS segmentation path and improving the mask representation.

## Standing Decisions

1. Multiple-person support is required.
2. macOS-specific optimizations are allowed and preferred.
3. The main success metric is high-quality real-time person masking that improves the final lensing effect.
4. Profiling and validation should accompany each substantial improvement.

## Current Status Summary

Accepted and working:

1. Phase 1 baseline profiling and characterization
2. Phase 2 macOS Vision-based segmentation path
3. Phase 3 soft-mask retention and hysteresis cleanup
4. Phase 4 image-guided boundary refinement
5. Phase 5 adaptive temporal stabilization

Attempted but not yet accepted:

1. Phase 6 ROI-based acceleration

Why Phase 6 is deferred:

1. The first ROI implementation introduced severe visible strobing and mask instability.
2. The failure mode was consistent with unstable switching between full-frame reacquisition and incomplete ROI updates.
3. ROI remains important for any future 60 FPS target, but it needs a more robust design before it can be enabled.

Current stable baseline:

1. Full-frame Vision segmentation at the configured working resolution
2. Soft-mask refinement with image-guided cleanup
3. Adaptive temporal smoothing
4. ROI acceleration disabled
5. Guidance-frame pairing on the Apple path so edge-aware cleanup uses the same frame that produced the Vision mask
6. Border-aware small-component retention to reduce clipped-person flicker at the camera edges

Observed stable performance envelope so far:

1. `person-mask-vision-request` is roughly `28-30 ms`
2. `person-mask-vision` is roughly `30-32 ms`
3. total `person-mask` is roughly `33-35 ms`
4. the current pipeline is viable around `30 FPS`
5. the current pipeline is not yet suitable for `60 FPS` without reducing Vision request cost further

Session restart stability:

1. Restarting a session while Vision segmentation was active could crash due to in-flight native-frame work during teardown.
2. The current implementation now explicitly begins segmentation-worker shutdown, disables new work, clears pending frame queues, and disconnects frame-routing connections before thread teardown.

## Refined Remaining Stages

### Next Active Stage: Refine How The Mask Drives Lensing

Goals:

1. Improve final visual quality without requiring a substantially better detector
2. Test whether a softened mass map produces a better result than the current near-binary silhouette
3. Reduce harsh discontinuities in the lensing effect at person boundaries

Planned work:

1. Compare binary mask vs softened mass map as lensing input
2. Try light blur or distance-like weighting before the lensing stage
3. Measure both perceived quality and any extra compute cost

Color mode relevance:

1. High. This stage should benefit color mode almost directly, because both modes eventually produce a mask that drives the same lensing pipeline.
2. If a softened mass map improves person mode, it is very likely worth testing against color mode too.

### Deferred But Important: Revisit ROI-Based Acceleration

The next ROI attempt should include:

1. ROI derived from a soft, stable probability field rather than the final binary mask alone
2. Strong overlap and inertia constraints between consecutive ROIs
3. Confidence-based fallback to full-frame when ROI confidence degrades
4. Multi-person-aware ROI union logic
5. A reacquisition path that cannot visibly strobe against the normal path

Color mode relevance:

1. Medium.
2. Color mode already has a form of region-focused tracking, so the exact Vision ROI design does not transfer directly.
3. The general ideas do transfer: ROI inertia, confidence-gated fallback, and anti-strobe reacquisition logic.

### Then: Add Runtime Quality Modes

Suggested modes:

1. `Fast`
2. `Balanced`
3. `High Quality`

These should control:

1. Vision quality level
2. Working segmentation resolution
3. Temporal aggressiveness
4. Refinement strength
5. Whether future ROI acceleration is enabled

Color mode relevance:

1. High.
2. Color mode should probably expose the same user-facing performance tiers, even if the underlying controls differ.
3. For color mode, those tiers might map to different mask smoothing, blob tracking aggressiveness, cleanup strength, and any future lensing-mass softening.
