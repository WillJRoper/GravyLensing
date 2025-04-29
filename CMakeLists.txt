# Minimum required CMake version and project definition
cmake_minimum_required(VERSION 3.10)
project(GravyLensing LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# FFTW3 Configuration (single-precision + threads)

# Path to FFTW installation (adjust as needed). For Homebrew on macOS ARM:
set(FFTW3_ROOT "/opt/homebrew/Cellar/fftw/3.3.10_2")
include_directories("${FFTW3_ROOT}/include")

# Locate the core single-precision FFTW library (libfftw3f)
find_library(FFTW3F_LIB
  NAMES fftw3f
  HINTS "${FFTW3_ROOT}/lib"
  NO_DEFAULT_PATH
)
if(NOT FFTW3F_LIB)
  message(FATAL_ERROR "Could not locate libfftw3f in ${FFTW3_ROOT}/lib")
endif()

# Locate the threaded FFTW wrapper (libfftw3f_threads)
find_library(FFTW3F_THREADS_LIB
  NAMES fftw3f_threads
  HINTS "${FFTW3_ROOT}/lib"
  NO_DEFAULT_PATH
)
if(NOT FFTW3F_THREADS_LIB)
  message(FATAL_ERROR "Could not locate libfftw3f_threads in ${FFTW3_ROOT}/lib")
endif()

# Combine libraries for linking
set(FFTW3_LIBRARIES
  ${FFTW3F_LIB}
  ${FFTW3F_THREADS_LIB}
)

# ============================================================================
# Other dependencies

# OpenCV for image processing
find_package(OpenCV REQUIRED)
# Portable threading library
find_package(Threads REQUIRED)
# LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)
# Qt6 Widgets for GUI
find_package(Qt6 COMPONENTS Widgets REQUIRED)

# ============================================================================
# Qt Configuration

# Enable automatic processing of Qt MOC for Q_OBJECT macros
set(CMAKE_AUTOMOC ON)
# Disable AUTOMOC for UIC and RCC (enable if using .ui or .qrc files)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

# ============================================================================
# Include directories

# Project source directory
include_directories("${CMAKE_SOURCE_DIR}/src")
# Third-party includes
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${Qt6Widgets_INCLUDE_DIRS}
)

# ============================================================================
# Source files

# List all source files under src/ directory
set(SOURCES
  src/main.cpp
  src/lens_mask.cpp
  src/viewport.cpp
  src/cam_feed.cpp
)

# ============================================================================
# Build target

# Define the executable target
add_executable(gravy_lens ${SOURCES})

# Place build output (binaries) one level above the build directory
set_target_properties(gravy_lens PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
)

# ============================================================================
# Linker settings

# Link against required libraries
target_link_libraries(gravy_lens PRIVATE
  ${OpenCV_LIBS}
  Threads::Threads
  ${FFTW3_LIBRARIES}
  ${TORCH_LIBRARIES}
  Qt6::Widgets
)

# Ensure that Qt exported symbols are available on all platforms
set_property(TARGET gravy_lens PROPERTY ENABLE_EXPORTS ON)

# ============================================================================
# Installation

# Install the executable into the 'bin' directory of the install prefix
install(TARGETS gravy_lens RUNTIME DESTINATION bin)
