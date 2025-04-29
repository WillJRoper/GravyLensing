# Minimum required CMake version and project definition
cmake_minimum_required(VERSION 3.10)
project(GravyLensing LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# OpenMP Configuration
# Find and enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found. Please install a compiler with OpenMP support.")
endif()

# ============================================================================
# FFTW3 Configuration (single-precision + threads)

# Allow user to specify FFTW3 installation prefix (e.g., /usr/local, /opt/homebrew)
option(FFTW3_ROOT "Path to FFTW3 installation prefix" "")
if(FFTW3_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${FFTW3_ROOT}")
endif()

# Find FFTW3 headers
find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  PATHS
    ${FFTW3_ROOT}/include
    /usr/local/include
    /opt/local/include
    /usr/include
)

# Find single-precision FFTW library
find_library(FFTW3F_LIB
  NAMES fftw3f
  PATHS
    ${FFTW3_ROOT}/lib
    /usr/local/lib
    /opt/local/lib
    /usr/lib
)

# Find threaded FFTW wrapper library
find_library(FFTW3F_THREADS_LIB
  NAMES fftw3f_threads
  PATHS
    ${FFTW3_ROOT}/lib
    /usr/local/lib
    /opt/local/lib
    /usr/lib
)

# Error out if any component is missing
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3F_LIB OR NOT FFTW3F_THREADS_LIB)
  message(FATAL_ERROR
    "Could not locate FFTW3 (single-precision + threads).\n"
    "Please install FFTW3 or set FFTW3_ROOT to its installation prefix."
  )
endif()

# Include FFTW headers and bundle libraries
include_directories("${FFTW3_INCLUDE_DIR}")
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
  src/backgrounds.cpp
)

# ============================================================================
# Build target

# Define the executable target
add_executable(gravy_lens ${SOURCES})

# Link OpenMP target (automatically adds appropriate compile & link flags)
target_link_libraries(gravy_lens PRIVATE OpenMP::OpenMP_CXX)

# Disable Qt's SIGNAL/SLOT macros so we don't break other libraries
target_compile_definitions(gravy_lens
  PRIVATE
    QT_NO_KEYWORDS
)

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
