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
- **Person segmentation mode**: Uses native Vision on macOS and TorchScript
  models (LR-ASPP or DeepLabV3) on Linux.
- **FFT-based lensing**: Applies gravitational deflection to background images
  based on the generated mask.
- **Multi-threaded**: OpenMP and threaded FFTW3 plans keep all pipeline stages
  concurrent.
- **Qt6 GUI**: Lensed output with an optional 2×2 diagnostic grid.
- **Background cycling**: Discovers supported images in the selected directory;
  switch with arrow keys, the menu, or auto-cycle.
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
- **libtorch** — PyTorch C++ API (≥ 2.0, Linux only)
- **Python 3.8+** — only for the optional model-generation script and the
  standalone Python example

macOS additionally links these system frameworks (no manual install needed):

- AVFoundation, CoreMedia, CoreVideo — camera capture
- Metal, MetalPerformanceShaders, Foundation — GPU acceleration

The distributed macOS app requires Apple Silicon and macOS 14 or newer.

## Installation

### macOS

Apple Silicon users can download the latest DMG from
[GitHub Releases](https://github.com/WillJRoper/gravy-lensing/releases), drag
**GravyLensing** to **Applications**, and open it. No Homebrew, Terminal, or
separate dependencies are required.

### Build from source

#### Clone

```bash
git clone https://github.com/WillJRoper/gravy-lensing.git
cd gravy-lensing
```

#### Dependencies

##### macOS (Homebrew)

```bash
brew install cmake fftw libomp opencv qt
```

`libomp` is required because AppleClang does not ship OpenMP by default.

##### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install cmake build-essential libfftw3-dev libfftw3-single3 \
  libopencv-dev qt6-base-dev python3 python3-venv python3-pip
```

##### libtorch (Linux only)

Download libtorch from [pytorch.org](https://pytorch.org/). Pass its path as
`CMAKE_PREFIX_PATH` during configuration.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch"
cmake --build build --config Release
```

On macOS:

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/libomp" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If FFTW3 is installed in a non-standard location, add `-DFFTW3_ROOT=/path/to/fftw3`.

On macOS, CMake creates `build/GravyLensing.app`. On Linux, the executable
`gravy_lens` is placed in the project root.

### Build options

| Flag | Default | Description |
|------|---------|-------------|
| `-DENABLE_PROFILING=ON` | OFF | Periodic `[Perf]` log lines showing average ms and fps per pipeline stage |
| `-DBUILD_TESTS=OFF` | ON | Skip building the unit-test binary |

## Segmentation models

Linux builds use the default model at
`models/lraspp_torchscript-traced_float32_512_512.pt`, which is what the app
uses for fresh Linux installs in Person mode. macOS uses native Vision and does
not require a model.

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

The guided Session Setup opens with remembered choices for subject detection,
camera, quality, region, and backgrounds. The recommended defaults require no
configuration; click **Start Session**. Use **Advanced Settings...** for full
control.

### CLI arguments

All flags are optional and seed the startup dialog. Omitted values use the last
saved session setting. Boolean flags accept an explicit `--no-` counterpart.

```
Usage: ./gravy_lens [options]

Options:
  -n, --nthreads <n>              Override automatic CPU thread allocation.
  -s, --strength <f>              Lens strength multiplier (default 4.0).
  -f, --softening <f>             Kernel softening radius in px (default 50.0).
  -m, --modelSize <n>             Segmentation model input size (default 512).
  -d, --deviceIndex <n>           Camera device index (default 0).
  --fps, --frameRate <n>           Target camera frame rate (default 30).
  -g, --debugGrid                 Show 2×2 diagnostic grid at start.
  --no-debugGrid                  Force the debug grid off.
  -p, --padFactor <n>             FFT padding multiplier (default 2).
  --mp, --modelPath <path>        TorchScript model path (Linux only).
  -t, --temporalSmooth <f>        Mask temporal blending factor (default 0.25).
  --personSensitivity <n>         Person sensitivity, 0–100 (default 50).
  --lr, --lowerRes <f>            Resolution scale for lensing, 0.1–1.0 (default 0.5).
  --quality, --qualityMode <mode>  fast, balanced, high, or custom.
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
- Only settings for the selected detection mode are shown.
- Recommended presets hide technical controls; **Custom** reveals them.
- Camera and background choices are validated directly in the dialog.
- **Restore Defaults** resets every control to shipped defaults.
- Colour and ROI selections persist across session restarts.

### Mask modes

**Person (AI segmentation)** — native Vision on macOS; TorchScript acceleration
where available on Linux.

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
| Previous / next background | `Left` / `Right` | View > Background |
| Open session settings | `Cmd+,` | Session > Session Settings... |
| Quit | `Esc` / `Cmd+Q` | File > Quit |

### Custom backgrounds

The macOS app includes a default set of backgrounds. To use your own, open
**Session > Session Settings...**, choose a backgrounds directory, and restart
the session. Supported images in that directory are discovered automatically;
use the left and right arrow keys to move through them. **Restore Defaults**
switches back to the packaged backgrounds.

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
