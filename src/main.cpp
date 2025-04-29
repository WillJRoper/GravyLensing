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

// Standard includes
#include <QApplication>
#include <QCoreApplication>
#include <fftw3.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Local includes
#include "backgrounds.hpp"
#include "cam_feed.hpp"
#include "cmd_parser.hpp"
#include "lens_mask.hpp"
#include "viewport.hpp"

// Define the path to the background images (this is constant)
const std::string bgDir = "backgrounds/";

int main(int argc, char **argv) {

  // Set up the Qt application itself
  QApplication app(argc, argv);

  // Set up and parse the command line options
  CommandLineOptions opts = CommandLineOptions::parse(app);

  // Unpack the options
  int nthreads = opts.nthreads;
  float strength = opts.strength;
  float softening = opts.softening;
  int maskScale = opts.maskScale;
  int deviceIndex = opts.deviceIndex;

  // Use threaded FFTW
  fftwf_init_threads();
  fftwf_plan_with_nthreads(nthreads);

  // Backgrounds are stored in the backgrounds/ directory at the root of the
  // repository, grab all the backgrounds and store their paths in a vector
  std::vector<std::string> bgPaths;

  // Load the background image
  Backgrounds backgrounds(bgDir);
  if (!backgrounds.load()) {
    std::cerr << "No images found in " << bgDir << "\n";
    return -1;
  }

  // Get an instance of the camera feed and
  CameraFeed *camFeed =
      new CameraFeed(0, backgrounds.cols(), backgrounds.rows());

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
  vp.setLens(&lens);
  vp.setBackgroundImages(&backgrounds);
  vp.setBackground(backgrounds.current());
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
    lens.applyLensing(backgrounds.current());

    // Push into the UI
    {
      std::lock_guard lk(camFeed->frameMutex_);
      vp.setImage(camFeed->latestFrame_); // top-left
    }

    {
      std::lock_guard lk(lens.maskMutex_);
      vp.setMask(lens.latestMask_); // top-right
    }

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
