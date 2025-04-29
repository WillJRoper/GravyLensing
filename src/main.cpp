/**
 * @file main.cpp
 *
 * @brief Main entry point for the GravyLensing application.
 *
 * This file is part of GravyLensing, a real-time gravitational lensing
 * simulation.
 *
 * GravyLensing is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GravyLensing is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GravyLensing. If not, see <http://www.gnu.org/licenses/>.
 */
#include <fftw3.h>

#include "cam_feed.hpp"
#include "lens_mask.hpp"
#include "viewport.hpp"
#include <QApplication>
#include <QCoreApplication>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {

  // Set up the Qt application itself
  QApplication app(argc, argv);

  if (argc < 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " <background_image> <nthreads> Optional: <strength> <softening> "
           "<maskScale> <deviceIndex>\n";
    return -1;
  }

  // Unpack the command line arguments
  std::string bgPath = argv[1];
  int nthreads = std::stoi(argv[2]);
  float strength = 0.1f;
  if (argc > 3) {
    strength = std::stof(argv[3]);
  }
  float softening = 30.0f;
  if (argc > 4) {
    softening = std::stof(argv[4]);
  }
  int maskScale = 4;
  if (argc > 5) {
    maskScale = std::stoi(argv[5]);
  }
  int deviceIndex = 0;
  if (argc > 6) {
    deviceIndex = std::stoi(argv[6]);
  }

  // Make sure we have more than 2 threads
  if (nthreads < 2) {
    std::cerr << "Need at least 2 threads, one for the main loop and one for "
                 "detection.\n";
    return -1;
  }

  // Use threaded FFTW
  fftwf_init_threads();
  fftwf_plan_with_nthreads(nthreads);

  // Load the background image
  cv::Mat background = cv::imread(bgPath, cv::IMREAD_COLOR);
  if (background.empty()) {
    std::cerr << "Failed to load background image: " << bgPath << "\n";
    return -1;
  }

  // Get an instance of the camera feed and
  CameraFeed *camFeed = new CameraFeed(0, background.cols, background.rows);

  // Set up LensMask
  const std::string modelPath =
      "/Users/willroper/Miscellaneous/gravy_lensing/models/"
      "deeplabv3_mobilenet_v3_large.pt";
  LensMask lens(camFeed, softening,
                /*padFactor=*/2, strength, modelPath, maskScale);

  // Build the lensing kernels we will use
  lens.buildKernels();

  // Set up ViewPort
  ViewPort vp;
  vp.setWindowTitle("GravyLensing Demo");
  // vp.showGridView();
  vp.showLensedView();
  vp.show(); // show the main window

  // We'll do the detection on a separate thread, so start it now
  lens.startAsyncSegmentation();

  // Main capture+render loop
  while (vp.isVisible()) {
    // Capture a new frame
    if (!camFeed->captureFrame()) {
      std::cerr << "Webcam frame failed, exiting.\n";
      break;
    }

    // Segment & lens
    lens.applyLensing(background);

    // Push into the UI
    {
      std::lock_guard lk(camFeed->frameMutex_);
      vp.setImage(camFeed->latestFrame_); // top-left
    }

    {
      std::lock_guard lk(lens.maskMutex_);
      vp.setMask(lens.latestMask_); // top-right
    }
    vp.setBackground(background); // bottom-left

    if (lens.newLensedImageReady_) {
      std::lock_guard lk(lens.lensedMutex_);
      vp.setLens(lens.latestLensed_); // bottom-right
    }

    // Process Qt events (gives you windowing, input, etc.)
    QCoreApplication::processEvents();
  }

  // Stop the segmentation thread
  lens.stopAsyncSegmentation();

  return 0;
}
