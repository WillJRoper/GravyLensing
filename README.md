# GravyLensing

A real-time gravitational lensing demo application written in C++ and supported by the Goodwood Festival Of Speed Future Lab.

Here is an example of the debug mode showing the mask overlaying me awkwardly sat at my desk along with the lensed and unlensed background.

![SCR-20250429-qemj](https://github.com/user-attachments/assets/39f96883-53b8-4d13-b399-3d390bf4328f)

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
Usage: ./gravy_lens [options]
GravyLensing applies a gravitational lensing effect to images based on people detected in a camera feed.

Options:
  -h, --help                                         Displays help on
                                                     commandline options.
  --help-all                                         Displays help, including
                                                     generic Qt options.
  -n, --nthreads <nthreads>                          Number of CPU threads used
                                                     in the calculation (must be
                                                     >= 2).
  -s, --strength <strength>                          Strength factor for the
                                                     lensing effect (float,
                                                     default=0.1).
  -f, --softening <softening>                        Softening radius in pixels
                                                     applied to the lensing
                                                     effect (float,
                                                     default=30.0).
  -m, --modelSize <modelSize>                        Segmentation model size,
                                                     bigger means more accurate
                                                     people but at the expense
                                                     of frame rate (int,
                                                     default=512).
  -d, --deviceIndex <deviceIndex>                    Device index, i.e. which
                                                     camera to use (int,
                                                     default=0).
  -g, --debugGrid                                    Show a debugging grid with
                                                     the camera feed, mask, and
                                                     lensed image.
  -p, --padFactor <padFactor>                        Padding factor for FFT
                                                     (int, default=2).
  --mp, --modelPath <modelPath>                      Path to the segmentation
                                                     model (string).
  -t, --temporalSmooth <temporalSmooth>              Temporal frame smoothing
                                                     factor, i.e. how much of
                                                     previous frames is used to
                                                     smooth out temporal
                                                     flucations in the person
                                                     detection mask (float,
                                                     default=0.25).
  --lr, --lowerRes <lowerRes>                        Lower resolution factor
                                                     for the lensing effect
                                                     (float, default=1.0).
  --sb, --secondsPerBackground <secondsPerBackground Seconds per background
  >                                                  image, if -1 then
                                                     background images are
                                                     selected through the 0-9
                                                     keys (int, default=-1).
  --di, --distortInside                              Distort inside the mask?
  --flip                                             Flip the camera feed
                                                     horizontally?
  --roi, --selectROI                                 Select a region of
                                                     interest (ROI) in the
                                                     camera feed to apply the
                                                     lensing effect. If not set,
                                                     the full frame is used.
```

- Place background images (up to 10) in the `backgrounds/` directory.
- Use 0-9 keys to cycle through backgrounds.
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
