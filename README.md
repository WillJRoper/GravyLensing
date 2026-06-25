# GravyLensing

A real-time gravitational lensing demo application written in C++ and supported by the Goodwood Festival Of Speed Future Lab.

Here is an example of the debug mode showing the mask overlaying me awkwardly sat at my desk along with the lensed and unlensed background.

![SCR-20250429-qemj](https://github.com/user-attachments/assets/39f96883-53b8-4d13-b399-3d390bf4328f)

## Features

- **Live camera input**: Captures webcam feed and segments the person in real time.
- **Fixed Color Key mode**: Chroma-key style colour masking with a user-selected HSV target.
- **Tracked Color Blob mode** (advanced): Connected-component blob tracking for more selective masking.
- **Gravitational lens effect**: Applies FFT-based deflection to background images based on person mask or colour key.
- **Multi-threaded**: Uses OpenMP and FFTW3 threaded plans for high performance.
- **Qt6 GUI**: Displays the lensed output using Qt6 (with an optional debugging view).
- **Segmentation model**: Uses TorchScript-exported models for person mask extraction via MPS/GPU.
- **Background cycling**: Load up to 10 images from `backgrounds/` and switch via key presses or menu.
- **Session-driven settings**: All configuration is managed through persistent session settings with an explicit startup dialog. Live mode/debug-grid changes persist across restarts.

## Prerequisites

- **CMake** ≥ 3.10
- **C++ compiler** with OpenMP support (e.g., GCC, Clang)
- **FFTW3** (single precision + threads)
- **OpenCV**
- **Qt6** (Widgets)
- **libtorch** (PyTorch C++ API)
- **Threads** (C++ std threads)
- **Python 3.8+** (for the example script and model generation)

## Installation

### Clone the repository

```bash
git clone https://github.com/WillJRoper/gravy-lensing.git
cd gravy-lensing
```

### Dependencies

Install via your package manager (assuming you need everything):

#### Linux (Ubuntu/Debian)

- Install via:
  ```bash
  sudo apt update
  sudo apt install cmake build-essential libfftw3-dev libfftw3-single3 libopencv-dev qt6-base-dev python3 python3-venv python3-pip
  ```

#### macOS (Homebrew)

- Install via:

  ```bash
  brew update
  brew install cmake fftw libomp opencv qt python@3.9
  ```

  `libomp` is required for fresh CMake configures on macOS because AppleClang
  does not ship OpenMP support by default.

  If FFTW3 is installed in non-standard locations, you will need to set
  `FFTW3_ROOT` during configuration.

#### Installing libtorch

For libtorch, see their [installation instructions](https://pytorch.org/). You will need to pass the location of libtorch at configuration time (as shown next).

## Build with CMake

To build the release build:

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j$(nproc)
```

Note that you may need to point directly to FFTW if it is installed in a nonstandard location:

```bash
cmake -B build \
  -DFFTW3_ROOT=/path/to/fftw3 \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -- -j$(nproc)
```

On macOS with Homebrew and a separate local libtorch install, a working
configure can look like:

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/libomp;/path/to/libtorch" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The executable `gravy_lens` will then be placed in the project root.

### Optional: profiling

To enable runtime profiling logs:

```bash
cmake -B build -DENABLE_PROFILING=ON
```

This adds periodic `[Perf] ...` log lines showing average ms and fps for each pipeline stage.
Profiling is disabled by default and has zero runtime overhead when off.

## Generating Segmentation models

Before running GravyLensing you will need some segmentation models to detect people in the frame. The repository already includes a working default model at `models/lraspp_torchscript-traced_float32_512_512.pt`, which is what the app now uses for fresh installs.

However, we also provide a unified Python script, `get_models.py` (in the `models/` directory), to generate TorchScript for the C++ inference pipeline. It currently supports two backbones—DeepLabV3 and LR-ASPP—and four export formats.

**Install prerequisites**

To run this script you'll need to have some PyTorch packages installed.

```bash
pip install torch torchvision
```

**Usage**

```bash
python get_models.py \
  --model <deeplab|lraspp> \
  --format <torchscript-scripted|torchscript-traced|quantized|onnx> \
  [--device cpu|cuda] [--width W] [--height H]
```

- `--model`
  - `deeplab` DeepLabV3 MobileNetV3 Large
  - `lraspp` LR-ASPP MobileNetV3 Large
- `--format`
  - `torchscript-scripted` uses `torch.jit.script(...)`
  - `torchscript-traced` uses `torch.jit.trace(...)` with a fixed dummy shape
  - `quantized` dynamic int8 quantization + scripted export (best CPU latency)
  - `onnx` ONNX opset 14 with dynamic axes (batch, height, width)
- `--device` (default `cpu`) load the model on CPU or GPU
- `--width`, `--height` (dummy input spatial size; default `320x320`)

**Output**
The script always writes to:

```
models/<model>_model.<ext>
```

- `.pt` for TorchScript formats
- `.onnx` for the ONNX export

**Example**

```bash
# Generate a quantized LR-ASPP model for fastest CPU inference
python get_models.py \
  --model lraspp \
  --format quantized

# Generate a traced DeepLabV3 model on GPU
python get_models.py \
  --model deeplab \
  --format torchscript-traced \
  --device cuda
```

## Usage

### Quick start

Launch the app; the session setup dialog opens. Pick the mask source first, adjust the enabled section for that mode, then click `Start Session`.

```bash
./gravy_lens
```

### CLI arguments

CLI arguments seed the session setup dialog with initial values. All flags are optional; any omitted value uses the last saved session setting. For persisted booleans you can now force either state explicitly with `--flag` or `--no-flag`.

```
Usage: ./gravy_lens [options]

Options:
  -n, --nthreads <n>            CPU threads (must be >= 2).
  -s, --strength <f>            Lens strength factor (default 0.1).
  -f, --softening <f>           Softening radius in pixels (default 30.0).
  -m, --modelSize <n>           Segmentation model size (default 512).
  -d, --deviceIndex <n>         Camera device index (default 0).
  -g, --debugGrid               Show 2x2 diagnostic grid at start.
  -p, --padFactor <n>           FFT padding factor (default 2).
  --mp, --modelPath <path>      TorchScript model path.
  -t, --temporalSmooth <f>      Temporal smoothing factor (default 0.25).
  --lr, --lowerRes <f>          Resolution scale for lensing (default 1.0).
  --sb, --secondsPerBackground <n>  Seconds per background; -1 = manual (default -1).
  --di, --distortInside          Distort inside the mask as well.
  --no-distortInside             Force interior distortion off.
  --flip                         Mirror camera feed horizontally.
  --no-flip                      Force camera mirroring off.
  --roi, --selectROI            Open ROI selector on first session start.
  --no-selectROI                 Skip the startup ROI selector.
  --no-debugGrid                 Force the debug grid off.
```

### Example session

A tuned setup for running on a laptop:

```bash
./gravy_lens --nthreads 12 --modelSize 512 --mp models/lraspp_torchscript-traced_float32_512_512.pt --softening 50 --strength 4 --lowerRes 0.5 --secondsPerBackground 3 --flip --distortInside
```

Choose `Person` mode in the session setup dialog and click `Start Session`.

### Settings panel notes

- The dialog is scrollable and works on smaller laptop displays.
- `Person Detection` settings are enabled only when `Person (AI segmentation)` is selected.
- `Color Detection` settings are enabled only when `Color tracking` is selected.
- Picking a colour from the dialog swatch or from the live camera now persists correctly across session restarts.

### Mask modes

**Person (`Person (AI segmentation)`)** uses a TorchScript segmentation model and (where available) MPS / GPU acceleration. This is the smoothest mode.

**Color (`Color tracking`)** has two submodes, selected in the `Color Detection` section of the session settings:

- **Fixed Color Key** (default): Behaves like a chroma-key matte. Pixels within a fixed HSV tolerance range of the selected target colour produce a mask. This is the fastest, most stable colour mode.

- **Tracked Color Blob** (advanced): Uses connected-components and blob-continuity heuristics to track a specific coloured region across frames. This mode can lose the blob even when keyed pixels are present; useful when you want to track a single object rather than every instance of a colour.

### Colour target selection

In colour mode, after starting the session:

- Choose (or re-choose) the target with `Shift+S` or `File > Select Color...`.
- The colour picker opens on the current camera frame. Click the target object, or press `Esc` / `c` to cancel.
- The selected target and its tolerances persist across session restarts.

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

To change pipeline settings mid-session:

1. Open `Session > Session Settings...`
2. Edit configuration
3. Click `Restart Session`

The current colour target and ROI are preserved across the restart where possible.

### Color mode tips

Color mode works best when the tracked object is strongly coloured, the background does not contain similar colours, and the lighting stays fairly stable. You do not need a segmentation model when running in colour mode.

When you pick a colour from the live camera, the app now carries the measured HSV spread forward as the starting tolerance range instead of dropping back to the fixed defaults immediately.

## Python Example

A simple self-contained Python demo is provided in `python_example.py`. This example implements some of the functionality of the C++ but with all the performance baggage you'd expect from Python. To run:

```bash
pip install torch torchvision opencv-python numpy
python python_example.py
```

This script:

1. Captures your webcam (`cv2.VideoCapture(0)`).
2. Loads a background image from `backgrounds/` by default.
3. Uses the same DeepLabV3 model for segmentation.
4. Applies half-resolution FFT lensing and displays the result.

## Contributing

Contributions, issues, and feature requests are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the GNU GPL-3.0 License. See [LICENSE](LICENSE) for details.
