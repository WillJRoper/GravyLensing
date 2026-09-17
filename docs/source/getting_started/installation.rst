Installation
============

macOS app (recommended)
-----------------------

Apple Silicon users do not need to build anything. Download the latest DMG from
`GitHub Releases <https://github.com/WillJRoper/gravy-lensing/releases>`_, drag
**GravyLensing** to **Applications**, and open it. No Homebrew, Terminal, or
separate dependencies are required.

The distributed app requires Apple Silicon and macOS 14 or newer. On first
launch macOS will ask for camera permission; the demo cannot run without it.

Building from source
--------------------

Requirements
~~~~~~~~~~~~

- **CMake** ≥ 3.10
- A **C++17 compiler** with OpenMP support
- **FFTW3**, single-precision plus the threaded wrapper (``fftw3f``,
  ``fftw3f_threads``)
- **OpenCV** ≥ 4
- **Qt6** — ``Core``, ``Gui``, ``Widgets``

On macOS the following system frameworks are linked automatically, with nothing
to install: AVFoundation, CoreMedia and CoreVideo for camera capture; Metal,
MetalPerformanceShaders and Foundation for GPU acceleration.

Clone and install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/WillJRoper/gravy-lensing.git
   cd gravy-lensing

On macOS, with Homebrew:

.. code-block:: bash

   brew install cmake fftw libomp opencv qt

``libomp`` is required because AppleClang does not ship OpenMP by default.

Build
~~~~~

.. code-block:: bash

   cmake -B build \
     -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/libomp" \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release

This produces ``build/GravyLensing.app``. If FFTW3 lives somewhere
non-standard, add ``-DFFTW3_ROOT=/path/to/fftw3``.

Build options
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Flag
     - Default
     - Description
   * - ``-DENABLE_PROFILING=ON``
     - OFF
     - Periodic ``[Perf]`` log lines showing average ms and fps per pipeline
       stage
   * - ``-DBUILD_TESTS=OFF``
     - ON
     - Skip building the unit-test binary
   * - ``-DGRAVY_VERSION=<version>``
     - ``1.0.0``
     - Version string baked into the bundle

Packaging a release
-------------------

The signed, distributable DMG is built by ``packaging/package_macos.sh``. See
`packaging/README.md
<https://github.com/WillJRoper/gravy-lensing/blob/main/packaging/README.md>`_
for what the script does, the environment variables it takes, and how CI uses
it.
