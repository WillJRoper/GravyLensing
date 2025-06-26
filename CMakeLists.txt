# Minimum required CMake version and project definition
cmake_minimum_required(VERSION 3.10)
project(GravyLensing LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define USE_MPS on macOS to enable MPS checks
if(APPLE)
  add_definitions(-DUSE_MPS)
endif()

# ----------------------------------------------------------------------------
# OpenMP Configuration
# ----------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found. Please install a compiler with OpenMP support.")
endif()

# ----------------------------------------------------------------------------
# FFTW3 Configuration (single-precision + threads)
# ----------------------------------------------------------------------------
option(FFTW3_ROOT "Path to FFTW3 installation prefix" "")
if(FFTW3_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${FFTW3_ROOT}")
endif()

find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h
  PATHS ${FFTW3_ROOT}/include /usr/local/include /opt/local/include /usr/include)
find_library(FFTW3F_LIB         NAMES fftw3f   PATHS ${FFTW3_ROOT}/lib /usr/local/lib /opt/local/lib /usr/lib)
find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads PATHS ${FFTW3_ROOT}/lib /usr/local/lib /opt/local/lib /usr/lib)

if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3F_LIB OR NOT FFTW3F_THREADS_LIB)
  message(FATAL_ERROR "Could not locate FFTW3 (single-precision + threads).\nPlease install FFTW3 or set FFTW3_ROOT to its installation prefix.")
endif()

include_directories("${FFTW3_INCLUDE_DIR}")
set(FFTW3_LIBRARIES ${FFTW3F_LIB} ${FFTW3F_THREADS_LIB})

# ----------------------------------------------------------------------------
# Other dependencies
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

# ----------------------------------------------------------------------------
# Qt Configuration
# ----------------------------------------------------------------------------
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

# ----------------------------------------------------------------------------
# Include directories
# ----------------------------------------------------------------------------
include_directories("${CMAKE_SOURCE_DIR}/src")
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# ----------------------------------------------------------------------------
# Source files
# ----------------------------------------------------------------------------
set(SOURCES
  src/main.cpp
  src/lensing_worker.cpp
  src/viewport.cpp
  src/cam_feed.cpp
  src/backgrounds.cpp
  src/segmentation_worker.cpp   # if you still use it elsewhere
  src/PixelBufferUtils.cpp
  src/VisionSegmentationWorker.cpp
  src/PersonSegmenter.mm
)

# ----------------------------------------------------------------------------
# Build target
# ----------------------------------------------------------------------------
add_executable(gravy_lens ${SOURCES})

# Compile ObjC++ for Vision bridge and pixel-buffer utils
set_source_files_properties(
  src/PersonSegmenter.mm
  src/PixelBufferUtils.cpp
  PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-x objective-c++"
)

# Place build output one level above build dir
set_target_properties(gravy_lens PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
)

# ----------------------------------------------------------------------------
# Link libraries
# ----------------------------------------------------------------------------
target_link_libraries(gravy_lens PRIVATE
  OpenMP::OpenMP_CXX
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  ${OpenCV_LIBS}
  Threads::Threads
  ${FFTW3_LIBRARIES}
  ${TORCH_LIBRARIES}
)

if(APPLE)
  find_library(VISION_FRAMEWORK    Vision)
  find_library(COREVIDEO_FRAMEWORK CoreVideo)
  find_library(COREMEDIA_FRAMEWORK CoreMedia)
  if(NOT VISION_FRAMEWORK OR NOT COREVIDEO_FRAMEWORK OR NOT COREMEDIA_FRAMEWORK)
    message(FATAL_ERROR "Could not find one of Vision, CoreVideo or CoreMedia frameworks")
  endif()
  target_link_libraries(gravy_lens PRIVATE
    ${VISION_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
    ${COREMEDIA_FRAMEWORK}
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
  )
endif()

# Ensure Qt symbols exported
set_property(TARGET gravy_lens PROPERTY ENABLE_EXPORTS ON)

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------
install(TARGETS gravy_lens RUNTIME DESTINATION bin)
