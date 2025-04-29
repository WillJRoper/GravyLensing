# gravy-lensing

A real-time gravitational lensing demo application written in C++.

## Features

- **Live camera input**: Captures webcam feed and segments the person in real time.
- **Gravitational lens effect**: Applies FFT-based deflection to background images based on person mask.
- **Multi-threaded**: Uses OpenMP and FFTW3 threaded plans for high performance.
- **Qt6 GUI**: Displays either lensed output or debug grid with mask, background, and camera feed.
- **Segmentation model**: Uses a TorchScript-exported DeepLabV3 MobileNet V3 model for person mask extraction.
- **Background cycling**: Load up to 10 images from `backgrounds/` and switch via key presses.
- **Python example**: Standalone script (`python_example.py`) demonstrating the lens effect in Python.

## Prerequisites

- **CMake** ≥ 3.10
- **C++ compiler** with OpenMP support (e.g., GCC, Clang)
- **FFTW3** (single precision + threads)
- **OpenCV**
- **Qt6** (Widgets)
- **libtorch** (PyTorch C++ API)
- **Threads** (C++ std threads)
- **Python 3.8+** (for the example script)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/WillJRoper/gravy-lensing.git
   cd gravy-lensing
   ```

2. **Prepare system dependencies**

   Install via your package manager (example for Ubuntu/Debian):
   ```bash
   sudo apt update
   sudo apt install cmake build-essential libfftw3-dev libfftw3-single3 libopencv-dev qt6-base-dev libtorch-dev
   ```

   - If FFTW3 or libtorch are installed in non-standard locations, set `FFTW3_ROOT` or `Torch_DIR` when invoking CMake.

3. **Build with CMake**

   ```bash
   mkdir build
   cd build
   cmake .. \
     -DFFTW3_ROOT=/path/to/fftw3 \
     -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/Qt6/lib/cmake" \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release -- -j$(nproc)
   ```

   The executable `gravy_lens` will be placed in the project root.

4. **Export the segmentation model**

   Create a TorchScript file from the pretrained DeepLabV3 MobileNet V3 model:

   ```python
   import torch
   from torchvision.models.segmentation import (
       deeplabv3_mobilenet_v3_large,
       DeepLabV3_MobileNet_V3_Large_Weights
   )

   # Load pretrained model
   model = deeplabv3_mobilenet_v3_large(
       weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
   )
   model.eval()

   # Script and save
   scripted = torch.jit.script(model)
   scripted.save("models/deeplabv3_mobilenet_v3_large.pt")
   ```

   Ensure the resulting `.pt` file is placed in a `models/` directory relative to the executable, or provide its path via `--modelPath`.

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
- Use left/right arrow keys in the GUI to cycle through backgrounds.
- Press `g` at launch (`--debugGrid`) to see the debug view.
- Press `q` or close the window to exit.

## Python Example

A self-contained Python demo is provided in `python_example.py`. To run:

```bash
pip install torch torchvision opencv-python numpy
python python_example.py
```

This script:
1. Captures your webcam (`cv2.VideoCapture(0)`).
2. Loads a background TIFF (update the `bg_path` variable).
3. Uses the same DeepLabV3 model for segmentation.
4. Applies half-resolution FFT lensing and displays the result.

## Project Structure

```
gravy-lensing/
├─ backgrounds/          # default folder for background images
├─ models/               # save your segmentation .pt model here
├─ src/                  # C++ source files
│  ├─ main.cpp
│  ├─ cmd_parser.hpp
│  ├─ backgrounds.hpp/.cpp
│  ├─ cam_feed.hpp/.cpp
│  ├─ lens_mask.hpp/.cpp
│  └─ viewport.hpp/.cpp
├─ CMakeLists.txt        # build configuration
├─ python_example.py     # Python demo script
├─ LICENSE               # GPL-3.0
└─ README.md             # this file
```

## Contributing

Contributions, issues, and feature requests are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the GNU GPL-3.0 License. See [LICENSE](LICENSE) for details.

