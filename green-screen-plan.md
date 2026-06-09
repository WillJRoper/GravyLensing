# Green Screen Mode Development Plan

## Goal

Add a new masking mode alongside the current person-segmentation path. In this mode, the user selects a target color at startup, and the app tracks that colored object over time to generate the mask used by the lensing pipeline.

This mode must be robust from the first implementation. It should not be a naive per-frame color threshold. It should behave like a small tracked-object system with guarded color adaptation and explicit loss/reacquisition behavior.

## High-Level Design

The existing app already has the right overall pipeline:

- `CameraFeed` emits frames
- a mask worker converts frames into a binary mask
- `LensingWorker` consumes that mask
- `ViewPort` optionally displays debug output

The cleanest integration is to add a new worker module:

- `src/color_mask.hpp`
- `src/color_mask.cpp`

This worker should mirror the current `SegmentationWorker` Qt interface as closely as possible so that `LensingWorker` stays unchanged.

## Non-Goals

- Do not replace the existing person segmentation mode.
- Do not weaken the current mask pipeline to fit the new mode.
- Do not start with raw RGB thresholding.
- Do not start with uncontrolled color-model updates.

## Success Criteria

The first usable version should:

- allow startup color selection from the camera feed
- track a distinctively colored object such as a ball
- continue working under moderate lighting drift
- avoid obvious drift onto the background
- survive short fragmentation or brief occlusion
- fail safely by freezing updates and attempting reacquisition
- produce a binary mask compatible with the existing lensing path

## Proposed User Experience

1. User launches the app with a new mask mode set to color.
2. The app opens the camera and captures an initial frame.
3. The user clicks the colored object to track.
4. The app samples a small patch around the click and initializes the color model.
5. During runtime, the tracker outputs a mask for that object.
6. In debug mode, the mask is shown in the existing grid.
7. If tracking confidence is lost, the tracker attempts reacquisition before giving up.

## Architecture Changes

### 1. Add a mask mode option

Files:

- `src/cmd_parser.hpp`
- `src/main.cpp`
- `README.md`

Add a new command-line option:

- `--maskMode person|color`

Default:

- `person`

Behavior:

- `person` keeps the current `SegmentationWorker` path
- `color` instantiates the new `ColorMaskWorker`

### 2. Add a dedicated color-mask worker

Files:

- `src/color_mask.hpp`
- `src/color_mask.cpp`

Public Qt shape should match the current segmentation worker closely:

- slot: `onFrame(const cv::Mat &frame)`
- slot: `onBackgroundChange(const cv::Mat &background)`
- signal: `maskReady(const cv::Mat &mask)`
- signal: `maskError(const std::string &error)` or reuse `segmentationError` naming if consistency is preferred

Internal responsibilities:

- initialize the target color model
- produce a per-frame mask
- maintain tracking state
- maintain guarded color adaptation
- manage loss and reacquisition state

### 3. Make `main.cpp` mode-aware

File:

- `src/main.cpp`

Current wiring assumes `SegmentationWorker` specifically. Refactor the setup so that:

- the person path still works unchanged in behavior
- the color path plugs into the same camera/background/lensing flow
- `LensingWorker` remains unaware of the mask source

This may require either:

- a small shared base interface for mask-producing workers, or
- a minimal branch in `main.cpp` that connects either worker type separately

Prefer the smaller change unless the wiring becomes awkward.

### 4. Reuse OpenCV startup interaction for color pick

Likely files:

- `src/cam_feed.cpp`
- `src/cam_feed.hpp`
- or keep the helper local to the new `color_mask` module if cleaner

The codebase already uses an OpenCV startup interaction for ROI selection. Reuse that interaction style for color picking rather than introducing a Qt-only picker flow.

## Tracking Algorithm

### 1. Initialization

At startup in color mode:

1. Capture a frame.
2. Display that frame in an OpenCV selection window.
3. Let the user click the target object.
4. Sample a patch around the click, not a single pixel.
5. Convert the patch to HSV.
6. Estimate the initial target color from the patch.
7. Reject initialization if the patch saturation is too low or the click is invalid.

Recommended defaults:

- patch size: `9x9` or `11x11`
- use median or trimmed mean rather than a raw mean if easy to implement
- require a minimum saturation floor

### 2. Frame Processing

For every frame:

1. Convert frame from BGR to HSV.
2. Threshold around the current target color model.
3. Apply cleanup operations.
4. Extract candidate components.
5. Score candidates against the tracking state.
6. Select the best blob.
7. If confidence is high enough, adapt the color model from the blob interior.
8. Emit the final binary mask.

### 3. Color Space

Use HSV, not RGB.

Reasoning:

- hue is more stable than raw RGB under brightness changes
- saturation can help reject gray shadows and highlights
- value can be tolerated with a wider band than hue

Implementation note:

- hue is circular and must not be averaged as a normal scalar near wraparound

### 4. Thresholding Strategy

Threshold using separate tolerances for:

- hue
- saturation
- value

Likely rules:

- hue uses a wrapped distance
- saturation has a floor to reject desaturated regions
- value should tolerate moderate changes in brightness

Start with sensible defaults in code rather than too many user-exposed flags.

### 5. Morphology and Cleanup

Apply basic cleanup before component selection:

- optional light blur if needed
- morphology open to remove speckles
- morphology close to fill small holes

Keep kernels small and test-driven. Over-smoothing will damage smaller targets.

### 6. Candidate Extraction

Use one of:

- connected components
- contours

Each candidate should provide:

- area
- centroid
- bounding box
- optional shape metrics such as compactness or circularity

### 7. Blob Selection

Do not always choose the largest component.

Score candidates using a combination of:

- area plausibility
- distance to previous centroid
- overlap with the previous bounding region
- compactness/circularity if tracking a roughly ball-like object
- border contact penalty

The selected blob becomes the tracked object for that frame.

### 8. Confidence-Gated Running Average

The running average of the object color is required in v1, but it must be guarded.

Update source region:

- use an eroded interior of the selected blob
- never update from the full thresholded mask before blob selection

Update rules:

- compute observed HSV from the interior region
- update slowly using an exponential moving average
- update hue using circular averaging
- update saturation and value using a standard EMA

Only update when all confidence checks pass.

### 9. Confidence Checks For Adaptation

Freeze color-model updates when any of these are true:

- selected blob area is too small
- candidate score is too weak
- blob fragments badly
- blob touches frame borders heavily
- observed color is too far from the current model
- multiple candidates are nearly tied

This is critical to prevent drift onto the background.

### 10. Tracking State

Persist the following state inside `ColorMaskWorker`:

- current target HSV model
- previous centroid
- previous area
- previous bounding box
- current confidence score
- lost-frame counter
- last good mask
- initialization status

### 11. Loss and Reacquisition

When the tracker confidence drops:

1. Freeze color adaptation.
2. Prefer candidates near the last known position.
3. Optionally widen thresholds slightly for a short grace period.
4. Keep trying to reacquire using the last locked color model.
5. If lost for too long, drop to a broader search state.

Important principle:

- losing the object should not immediately mutate the color model

### 12. Failure Policy

If no good target is found:

- emit either the last good mask for a short grace window, or
- emit an empty mask once the loss threshold is exceeded

This behavior should be chosen deliberately and documented. For robustness, a short grace window is likely better than instant disappearance.

## File-By-File Implementation Plan

### Step 1. Extend CLI parsing

File:

- `src/cmd_parser.hpp`

Tasks:

- add `maskMode` to `CommandLineOptions`
- parse `--maskMode`
- validate allowed values
- keep default as `person`
- update help text to describe both modes

### Step 2. Add color-pick startup helper

Files:

- likely `src/color_mask.cpp`
- possibly shared helper placement in `src/cam_feed.cpp` if reuse becomes useful

Tasks:

- capture an initialization frame
- show a preview window
- handle mouse click selection
- sample a patch around the click
- return an initial color model
- fail cleanly if selection is cancelled or invalid

### Step 3. Add `ColorMaskWorker` header

File:

- `src/color_mask.hpp`

Tasks:

- define the QObject worker class
- expose Qt slots/signals matching the existing mask pipeline
- define persistent tracking state
- define helper types if needed for color model and tracked candidate

Keep helper types minimal and local to the module.

### Step 4. Implement initialization and state setup

File:

- `src/color_mask.cpp`

Tasks:

- initialize Mats and buffers
- store model parameters and thresholds
- initialize color model from the startup selection
- mark the worker ready only after successful selection

### Step 5. Implement per-frame thresholding and cleanup

File:

- `src/color_mask.cpp`

Tasks:

- convert frame to HSV
- create a binary candidate mask from wrapped hue distance and S/V thresholds
- apply morphology
- prepare the mask for component extraction

### Step 6. Implement component extraction and scoring

File:

- `src/color_mask.cpp`

Tasks:

- identify connected components or contours
- compute area, centroid, bounding box, and optional shape metrics
- score candidates against prior state
- select the best candidate or declare the frame low-confidence

### Step 7. Implement guarded color adaptation

File:

- `src/color_mask.cpp`

Tasks:

- erode the selected blob to get a trusted interior
- compute observed HSV inside that region
- update hue with circular EMA
- update S/V with standard EMA
- block updates when confidence checks fail

### Step 8. Implement loss and reacquisition logic

File:

- `src/color_mask.cpp`

Tasks:

- maintain lost-frame counter
- freeze adaptation during uncertainty
- keep short-term positional preference
- optionally widen thresholds during reacquisition
- fall back to empty or stale mask after too many failed frames

### Step 9. Implement geometry handling

Files:

- `src/color_mask.hpp`
- `src/color_mask.cpp`

Tasks:

- mirror the `onBackgroundChange` behavior used by `SegmentationWorker`
- size the output mask to the lensing resolution expected by `LensingWorker`
- ensure the emitted mask shape matches `LensingWorker::onMask`

### Step 10. Wire color mode into `main.cpp`

File:

- `src/main.cpp`

Tasks:

- instantiate the correct mask worker based on `opts.maskMode`
- connect camera frames to that worker
- connect mask output to `LensingWorker`
- connect background changes to that worker
- connect debug display if enabled
- connect error reporting

Keep the person mode behavior unchanged.

### Step 11. Update build system

File:

- `CMakeLists.txt`

Tasks:

- add `src/color_mask.cpp` to the sources list
- verify no new dependencies are required beyond OpenCV/Qt already in use

### Step 12. Update documentation

File:

- `README.md`

Tasks:

- document the new color mask mode
- document the startup click-to-select behavior
- document limitations and recommended object/background choices
- add a sample command line

## Suggested Internal Defaults

These should start as code constants unless testing shows a strong need for CLI exposure:

- patch size for startup sample
- hue tolerance
- saturation floor
- value tolerance
- morphology kernel sizes
- EMA alpha for color updates
- erosion amount for trusted interior sampling
- minimum blob area
- lost-frame threshold
- reacquisition threshold widening

## Verification Plan

### Functional verification

Test all of the following:

1. startup selection succeeds on a strongly colored object
2. tracking remains stable while the object moves around frame center
3. tracking remains stable near image edges
4. moderate room lighting changes do not immediately break tracking
5. brief occlusion does not cause catastrophic drift
6. similarly colored background clutter does not easily steal the track
7. when tracking is lost, the tracker freezes adaptation and attempts reacquisition
8. switching back to `person` mode still works unchanged

### Debug verification

Use the existing debug grid to inspect:

- camera frame
- mask quality
- final lensed output

If needed later, add a temporary overlay view showing centroid or bounding box, but avoid broad UI changes in the first pass.

### Regression verification

After adding the new mode:

- build the project successfully
- verify that the default mode still uses person segmentation
- verify that background switching still works
- verify that ROI mode still behaves sensibly in combination with color mode if both are enabled

## Risks and Mitigations

### Risk: drift onto similarly colored background

Mitigation:

- guarded adaptation only from eroded blob interior
- freeze updates on low confidence
- use temporal association

### Risk: object highlights or shadows shift apparent color

Mitigation:

- use HSV rather than RGB
- update slowly
- avoid updating from low-saturation or low-confidence regions

### Risk: tracker latches onto wrong blob after occlusion

Mitigation:

- score by position continuity, not just size
- add border penalties and area plausibility checks
- use reacquisition state rather than immediate relock

### Risk: thresholds too brittle across cameras

Mitigation:

- choose conservative defaults
- expose tuning flags only after initial testing if really needed

### Risk: implementation complexity grows inside `main.cpp`

Mitigation:

- keep mask-source branching localized
- only introduce a shared mask-worker abstraction if the duplication becomes messy

## Milestones

### Milestone 1. Plumbing

- CLI option added
- `ColorMaskWorker` skeleton added
- app can boot in `color` mode and emit a mask-shaped output

### Milestone 2. Core tracking

- startup color selection works
- HSV thresholding and morphology work
- blob selection works on a colored object

### Milestone 3. Robust adaptation

- guarded running average implemented
- drift resistance improved
- loss/reacquisition logic working

### Milestone 4. Documentation and tuning

- README updated
- defaults tuned from real usage
- person mode regression-checked

## Immediate Next Steps

1. Add `--maskMode person|color` to the CLI parser.
2. Decide whether the startup color-pick helper lives in `color_mask.cpp` or a shared camera helper.
3. Define the `ColorMaskWorker` state layout in the header.
4. Implement the startup color selection path before the main tracking loop.
5. Implement thresholding, candidate scoring, and guarded adaptation in that order.
