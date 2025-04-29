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
#include <QTimer>
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
  int padFactor = opts.padFactor;

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
  LensMask lens(camFeed, softening, padFactor, strength, modelPath, maskScale);

  // Set up ViewPort
  ViewPort vp;

  // Set the title of the window
  vp.setWindowTitle("GravyLensing Demo");

  // Attach pointers to the lens and backgrounds (we'll need them for
  // key press interactions: switching background etc.)
  vp.setLens(&lens);
  vp.setBackgroundImages(&backgrounds);

  // Set the initial image to the first background
  vp.setBackground(backgrounds.current());

  // Are we using the debug view or the "production" view with only the lensed
  // image?
  if (opts.debugGrid) {
    vp.showGridView();
  } else {
    vp.showLensedView();
  }

  // Show the main window
  vp.show();

  // We'll do the detection on a separate thread, so start it now
  lens.startAsyncSegmentation();

  // Use QTimer to drive capture
  QTimer *frameTimer = new QTimer(&vp);
  QObject::connect(frameTimer, &QTimer::timeout, [&]() {
    // Capture frame
    if (!camFeed->captureFrame()) {
      qWarning("Camera frame failed, stopping timer.");
      frameTimer->stop();
      return;
    }

    // Apply lensing
    lens.applyLensing(backgrounds.current(), nthreads);

    // Update viewport images
    vp.setImage(camFeed->latestFrame_);
    if (lens.newLensedImageReady_) {
      vp.setLens(lens.latestLensed_);
      if (opts.debugGrid) {
        vp.setMask(lens.latestMask_);
        vp.setBackground(backgrounds.current());
      }
    }
  });
  frameTimer->start(1000 / 60); // ~60 fps

  // Enter Qt event loop
  int ret = app.exec();

  // Clean up
  frameTimer->stop();
  lens.stopAsyncSegmentation();

  return 0;
}
