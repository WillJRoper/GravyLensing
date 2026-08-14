# Minimum required CMake version and project definition
cmake_minimum_required(VERSION 3.10)
set(GRAVY_VERSION "1.0.0" CACHE STRING "GravyLensing release version")
project(GravyLensing VERSION ${GRAVY_VERSION} LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "GravyLensing requires macOS because person detection uses Apple Vision.")
endif()

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_PROFILING "Enable lightweight runtime profiling logs" OFF)

if(ENABLE_PROFILING)
  add_definitions(-DGRAVY_ENABLE_PROFILING)
endif()

# Define USE_MPS on macOS to enable MPS checks
if(APPLE)
  add_definitions(-DUSE_MPS)
endif()

# ============================================================================
# OpenMP Configuration
# Homebrew installs libomp as a keg-only dependency on macOS, so help CMake
# find it during fresh configures.
if(APPLE)
  list(APPEND CMAKE_PREFIX_PATH
    "/opt/homebrew/opt/libomp"
    "/usr/local/opt/libomp"
  )
  find_library(HOMEBREW_LIBOMP
    NAMES omp
    PATHS
      /opt/homebrew/opt/libomp/lib
      /usr/local/opt/libomp/lib
    NO_DEFAULT_PATH
    REQUIRED
  )
  # OpenCV reaches this runtime through OpenBLAS. Linking LLVM's separate copy
  # as well causes OpenMP error #15 and aborts when processing starts.
  set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP}" CACHE FILEPATH
      "OpenMP runtime shared with Homebrew dependencies" FORCE)
endif()

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
if(APPLE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)
endif()
# Portable threading library
find_package(Threads REQUIRED)
# Qt6 Widgets for GUI
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

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
)

# ============================================================================
# Source files

# List all source files under src/ directory
set(SOURCES
  src/main.cpp
  src/color_mask.cpp
  src/segmentation_worker.cpp 
  src/lensing_worker.cpp
  src/viewport.cpp
  src/cam_feed.cpp
  src/backgrounds.cpp
  src/session_setup_dialog.cpp
  src/settings_dialog.cpp
)

if(APPLE)
  list(APPEND SOURCES src/metal_helper.mm)
  list(APPEND SOURCES src/avfoundation_camera.mm)
  list(APPEND SOURCES src/vision_segmentation_helper.mm)
  # ARC is required for the Vision helper because it uses autoreleased
  # ObjC objects (CIContext, VNRequest, etc.) that would become dangling
  # pointers without automatic retain/release.
  # The other .mm files use manual retain/release and must not use ARC.
  set_source_files_properties(src/vision_segmentation_helper.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# ============================================================================
# Build target

# Define the executable target
add_executable(gravy_lens ${SOURCES})

if(APPLE)
  file(GLOB BACKGROUND_RESOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/backgrounds/*.tiff")
  target_sources(gravy_lens PRIVATE ${BACKGROUND_RESOURCES})
  set_source_files_properties(${BACKGROUND_RESOURCES} PROPERTIES
    MACOSX_PACKAGE_LOCATION "Resources/backgrounds")
  target_sources(gravy_lens PRIVATE LICENSE)
  set_source_files_properties(LICENSE PROPERTIES
    MACOSX_PACKAGE_LOCATION "Resources")
endif()

# Link OpenMP target (automatically adds appropriate compile & link flags)
target_link_libraries(gravy_lens PRIVATE OpenMP::OpenMP_CXX)

if(APPLE)
  set_target_properties(gravy_lens PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.willroper.gravylensing"
    MACOSX_BUNDLE_BUNDLE_NAME "GravyLensing"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    OSX_ARCHITECTURES "arm64"
    OUTPUT_NAME "GravyLensing"
  )
else()
  set_target_properties(gravy_lens PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
  )
endif()

# ============================================================================
# Linker settings

target_link_libraries(gravy_lens PRIVATE
  Qt6::Core
  Qt6::Gui       
  Qt6::Widgets
  ${OpenCV_LIBS}
  Threads::Threads
  ${FFTW3_LIBRARIES}
)

if(APPLE)
  target_link_libraries(gravy_lens PRIVATE
    "-framework AVFoundation"
    "-framework Foundation"
    "-framework CoreMedia"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework CoreVideo"
    "-framework CoreImage"
    "-framework Vision"
    "-framework ImageIO"
  )
endif()

# Ensure that Qt exported symbols are available on all platforms
set_property(TARGET gravy_lens PROPERTY ENABLE_EXPORTS ON)

# ============================================================================
# Installation

# Install the executable into the platform's standard destination.
install(TARGETS gravy_lens BUNDLE DESTINATION . RUNTIME DESTINATION bin)

# ============================================================================
# Unit tests
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
  add_executable(test_color_mask
    src/color_mask.cpp
    tests/test_color_mask.cpp
  )

  if(APPLE)
    target_sources(test_color_mask PRIVATE src/metal_helper.mm)
    target_link_libraries(test_color_mask PRIVATE
      "-framework Foundation"
      "-framework Metal"
    )
  endif()

  target_link_libraries(test_color_mask PRIVATE
    Qt6::Core
    ${OpenCV_LIBS}
  )

  set_property(TARGET test_color_mask PROPERTY ENABLE_EXPORTS ON)

  set_target_properties(test_color_mask PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/.."
  )

  enable_testing()
  add_test(NAME color_mask COMMAND test_color_mask)

  add_executable(test_backgrounds
    src/backgrounds.cpp
    tests/test_backgrounds.cpp
  )
  target_link_libraries(test_backgrounds PRIVATE
    Qt6::Core
    ${OpenCV_LIBS}
  )
  add_test(NAME backgrounds COMMAND test_backgrounds)

  add_executable(test_settings tests/test_settings.cpp)
  target_link_libraries(test_settings PRIVATE Qt6::Core)
  add_test(NAME settings COMMAND test_settings)
endif()
