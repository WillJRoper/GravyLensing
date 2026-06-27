# GravyLensing

A real-time gravitational lensing demo written in C++, supported by the Goodwood
Festival Of Speed Future Lab.

![SCR-20250429-qemj](https://github.com/user-attachments/assets/39f96883-53b8-4d13-b399-3d390bf4328f)

## Features

- **Live camera input**: Captures webcam feed in real time (native AVFoundation on
  macOS, OpenCV `VideoCapture` on Linux).
- **Metal GPU acceleration** (macOS only): Offloads colour-key thresholding and
  lens-map construction to the GPU.
- **Fixed Color Key mode**: Chroma-key style HSV masking against a user-selected
  target colour.
- **Tracked Color Blob mode** (advanced): Connected-component blob tracking for
  selective single-object masking.
- **Person segmentation mode**: Uses TorchScript models (LR-ASPP or DeepLabV3)
  with MPS/GPU inference for person detection.
- **FFT-based lensing**: Applies gravitational deflection to background images
  based on the generated mask.
- **Multi-threaded**: OpenMP and threaded FFTW3 plans keep all pipeline stages
  concurrent.
- **Qt6 GUI**: Lensed output with an optional 2×2 diagnostic grid.
- **Background cycling**: Loads up to 10 images from `backgrounds/`; switch with
  `0`–`9` keys, a menu, or auto-cycle.
- **Session-driven settings**: All configuration is persisted through an
  explicit startup dialog. Live mode and debug-grid toggles survive restarts.
- **Coalescing frame delivery**: Workers accept only the most recently arrived
  frame, avoiding backlog buildup under load.

## Prerequisites

- **CMake** ≥ 3.10
- **C++17 compiler** with OpenMP support
- **FFTW3** — single-precision library + threaded wrapper (`fftw3f`,
  `fftw3f_threads`)
- **OpenCV** ≥ 4
- **Qt6** — `Core`, `Gui`, `Widgets`
- **libtorch** — PyTorch C++ API (≥ 2.0)
- **Python 3.8+** — only for the optional model-generation script and the
  standalone Python example

macOS additionally links these system frameworks (no manual install needed):

- AVFoundation, CoreMedia, CoreVideo — camera capture
- Metal, MetalPerformanceShaders, Foundation — GPU acceleration

## Installation

### Clone

```bash
git clone https://github.com/WillJRoper/gravy-lensing.git
cd gravy-lensing
```

### Dependencies

#### macOS (Homebrew)

```bash
brew install cmake fftw libomp opencv qt
```

`libomp` is required because AppleClang does not ship OpenMP by default.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install cmake build-essential libfftw3-dev libfftw3-single3 \
  libopencv-dev qt6-base-dev python3 python3-venv python3-pip
```

#### libtorch

Download libtorch from [pytorch.org](https://pytorch.org/). Pass its path as
`CMAKE_PREFIX_PATH` during configuration.

## Build

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On macOS with Homebrew libtorch:

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/libomp;/path/to/libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If FFTW3 is installed in a non-standard location, add `-DFFTW3_ROOT=/path/to/fftw3`.

The executable `gravy_lens` is placed in the project root.

### Build options

| Flag | Default | Description |
|------|---------|-------------|
| `-DENABLE_PROFILING=ON` | OFF | Periodic `[Perf]` log lines showing average ms and fps per pipeline stage |
| `-DBUILD_TESTS=OFF` | ON | Skip building the unit-test binary |

## Segmentation models

The repository ships a default model at
`models/lraspp_torchscript-traced_float32_512_512.pt`, which is what the app
uses for fresh installs in Person mode.

The script `models/get_models.py` can generate additional models:

```bash
pip install torch torchvision
python models/get_models.py --model lraspp --format quantized
```

Supported backbones: `deeplab`, `lraspp`.  
Supported formats: `torchscript-scripted`, `torchscript-traced`, `quantized`, `onnx`.

See `models/README` for the models already included.

## Usage

### Quick start

```bash
./gravy_lens
```

The session-setup dialog opens. Pick the mask source, adjust the relevant
section, then click **Start Session**.

### CLI arguments

All flags are optional and seed the startup dialog. Omitted values use the last
saved session setting. Boolean flags accept an explicit `--no-` counterpart.

```
Usage: ./gravy_lens [options]

Options:
  -n, --nthreads <n>              CPU threads (must be ≥ 2; default 12).
  -s, --strength <f>              Lens strength multiplier (default 4.0).
  -f, --softening <f>             Kernel softening radius in px (default 50.0).
  -m, --modelSize <n>             Segmentation model input size (default 512).
  -d, --deviceIndex <n>           Camera device index (default 0).
  -g, --debugGrid                 Show 2×2 diagnostic grid at start.
  --no-debugGrid                  Force the debug grid off.
  -p, --padFactor <n>             FFT padding multiplier (default 2).
  --mp, --modelPath <path>        TorchScript model path.
  -t, --temporalSmooth <f>        Mask temporal blending factor (default 0.25).
  --lr, --lowerRes <f>            Resolution scale for lensing, 0.1–1.0 (default 0.5).
  --sb, --secondsPerBackground <n> Seconds per background; -1 = manual (default -1).
  --di, --distortInside           Also lens the interior of the mask (default on).
  --no-distortInside              Force interior distortion off.
  --flip                          Mirror camera feed horizontally.
  --no-flip                       Force mirroring off.
  --roi, --selectROI              Open ROI selector on first session start.
  --no-selectROI                  Skip startup ROI selector.
```

### Example session

```bash
./gravy_lens
```

Choose **Person** mode and click **Start Session**.

### Settings panel

- The dialog is scrollable and works on smaller laptop displays.
- **Person Detection** controls are enabled only when Person mode is selected.
- **Color Detection** controls are enabled only when Color mode is selected.
- **Resolution scale** uses a slider alongside the spin box.
- **Restore Defaults** resets every control to shipped defaults.
- Colour and ROI selections persist across session restarts.

### Mask modes

**Person (AI segmentation)** — TorchScript model with MPS/GPU acceleration where
available.

**Color tracking** with two sub-modes:

- **Fixed Color Key** (default): Chroma-key matte from a user-picked HSV range.
  Fast and stable.
- **Tracked Color Blob** (advanced): Tracks a single connected-colour region
  across frames using blob-continuity heuristics.

### Colour target selection

In Color mode:

- Pick or re-pick a target with `Shift+S` or **File > Select Color...**.
- Click the target object in the OpenCV picker window, or press `Esc`/`c` to
  cancel.
- The measured HSV spread is used as the starting tolerance range.
- Colour mode does not require a segmentation model.

### During a session

| Action | Shortcut | Menu |
|--------|----------|------|
| Select / reselect colour | `Shift+S` | File > Select Color... |
| Select region of interest | `Shift+R` | File > Select Region... |
| Toggle debug grid | `Shift+D` | View > Debug Grid |
| Switch mask mode | `Shift+M` | View > Mask Mode |
| Switch background | `0`–`9` | View > Background |
| Open session settings | `Cmd+,` | Session > Session Settings... |
| Quit | `Esc` / `Cmd+Q` | File > Quit |

### Restarting a session

1. Open **Session > Session Settings...**
2. Edit the configuration.
3. Click **Restart Session**.

The current colour target and ROI are preserved across restarts where possible.

## Python example

`python_example.py` is a standalone Python demo with the same pipeline, but
without the performance of the C++ version.

```bash
pip install torch torchvision opencv-python numpy
python python_example.py
```

It loads a background from `backgrounds/` automatically.

## Contributing

Contributions, issues, and feature requests are welcome. Fork the repository and
submit a pull request.

## License

GNU GPL-3.0. See [LICENSE](LICENSE) for details.
