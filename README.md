# GravyLensing

A real-time gravitational lensing demo application written in C++.

## Features

- **Live camera input**: Captures webcam feed and segments the person in real time.
- **Gravitational lens effect**: Applies FFT-based deflection to background images based on person mask.
- **Multi-threaded**: Uses OpenMP and FFTW3 threaded plans for high performance.
- **Qt6 GUI**: Displays the lensed output using Qt6 (with an optional debugging view).
- **Segmentation model**: Uses a TorchScript-exported models for person mask extraction.
- **Background cycling**: Load up to 10 images from `backgrounds/` and switch via key presses.

## TODO:

- Optimise segementation step to remove bottleneck and smooth out small scale variations.
- Enable turning on and off of "lens" feed (i.e. the person) in output video feed.
- Utilise GPUs when available.
- Scalable lensing strength.

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
  brew install cmake fftw opencv qt python@3.9
  ```

  If FFTW3 is installed in non-standard locations, you will need to set `FFTW3_ROOT` during configuration.

#### Installing libtorch

For libtorch, see their [installation instructions](https://pytorch.org/). You will need to pass the location of libtorch at configuration time (as shown next).

## Build with CMake

   To build the release build:

   ```bash
   cmake -B build \
     -DCMAKE_PREFIX_PATH=/path/to/libtorch/ \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -- -j$(nproc)
   ```

   Note that you may need to point directly to FFTW if it is installed in a nonstandard location:

   ```bash
   cmake -B build \
     -DFFTW3_ROOT=/path/to/fftw3 \
     -DCMAKE_PREFIX_PATH=/path/to/libtorch/ \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -- -j$(nproc)
   ```

   The executable `gravy_lens` will then be placed in the project root.

## Generating Segementation models

Before running GravyLensing you will need some segmentation models to detect people in the frame. We provide a unified Python script, `get_models.py` (in the `models/` directory), to generate TorchScript or ONNX artifacts for the C++ inference pipeline. It currently supports two backbones—DeepLabV3 and LR-ASPP—and four export formats.

NOTE: ONNX is not yet supported in the C++ app but will be.

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
  - `deeplab` DeepLabV3 MobileNetV3 Large
  - `lraspp` LR-ASPP MobileNetV3 Large
- `--format`
  - `torchscript-scripted` uses `torch.jit.script(…)`
  - `torchscript-traced` uses `torch.jit.trace(…)` with a fixed dummy shape
  - `quantized` dynamic int8 quantization + scripted export (best CPU latency)
  - `onnx` ONNX opset 14 with dynamic axes (batch, height, width)
- `--device` (default `cpu`) load the model on CPU or GPU
- `--width`, `--height` (dummy input spatial size; default `320×320`)

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

```bash
./gravy_lens \
  --nthreads <int>       # number of FFTW threads (≥2)
  [--strength <float>]    # lens strength factor (default 0.1)
  [--softening <float>]   # softening radius (default 30.0)
  [--maskScale <int>]     # segmentation downscale factor (default 4)
  [--deviceIndex <int>]   # Torch device index (default 0)
  [--padFactor <int>]     # FFT padding factor (default 2)
  [--modelPath <string>]  # path to segmentation .pt model
  [--debugGrid]           # show mask & camera feed grid
```

- Place background images (up to 10) in the `backgrounds/` directory.
- Use 0-9 keys to cycle through backgrounds.
- Press `g` at launch (`--debugGrid`) to see the debug view
- Press `ESC` or close the window to exit.

## Python Example

A simple self-contained Python demo is provided in `python_example.py`. This example implements some of the functionality of the C++ but with all the performance baggage you'd expect from Python. To run:

```bash
pip install torch torchvision opencv-python numpy
python python_example.py
```

This script:

1. Captures your webcam (`cv2.VideoCapture(0)`).
2. Loads a background TIFF (update the `bg_path` variable).
3. Uses the same DeepLabV3 model for segmentation.
4. Applies half-resolution FFT lensing and displays the result.

## Contributing

Contributions, issues, and feature requests are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the GNU GPL-3.0 License. See [LICENSE](LICENSE) for details.
