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
#include <QElapsedTimer>
#include <QTimer>
#include <chrono>
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
  bool debugGrid = opts.debugGrid;
  const std::string modelPath = opts.modelPath;

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
  LensMask lens(camFeed, softening, padFactor, strength, modelPath, maskScale);

  // Set up ViewPort
  ViewPort vp;

  // Set the title of the window
  vp.setWindowTitle("Gravy Lensing");

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

  // Initialise the timer and FPS (we only use the latter for debugging)
  static QElapsedTimer fpsTimer;
  static int fpsFrameCount = 0;
  fpsTimer.start();

  // Use QTimer to drive capture
  QTimer *frameTimer = new QTimer(&vp);
  QObject::connect(frameTimer, &QTimer::timeout, [&]() {
    // Capture frame
    double capture_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (!camFeed->captureFrame()) {
      qWarning("Camera frame failed, stopping timer.");
      frameTimer->stop();
      return;
    }
    double capture_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Capture time: " << (capture_end - capture_start) / 1e6
              << " ms\n";

    double detect_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
#ifdef USE_MPS
    lens.detectPersonMaskGPU(nthreads);
#else
    lens.detectPersonMask();
#endif
    double detect_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Detection time: " << (detect_end - detect_start) / 1e6
              << " ms\n";

    // Apply lensing
    double lens_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    lens.applyLensing(backgrounds.current(), nthreads);
    double lens_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Lensing time: " << (detect_end - detect_start) / 1e6
              << " ms\n";

    // Update viewport images
    double update_start =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    vp.setLens(lens.latestLensed_);
    double update_end =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (debugGrid) {
      double db_update_start =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      vp.setImage(camFeed->latestFrame_);
      vp.setMask(lens.latestMask_);
      vp.setBackground(backgrounds.current());
      double db_update_end =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      std::cout << "Debug update time: "
                << (db_update_end - db_update_start) / 1e6 << " ms\n";

      // FPS measurement
      double fps_start =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      ++fpsFrameCount;
      qint64 elapsed = fpsTimer.elapsed(); // ms since start
      if (elapsed >= 1000) {               // once per second
        double fps = fpsFrameCount * 1000.0 / elapsed;
        qDebug("Approx FPS: %.1f", fps);
        // reset for next interval
        fpsTimer.restart();
        fpsFrameCount = 0;
      }
      double fps_end =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      std::cout << "FPS time: " << (fps_end - fps_start) / 1e6 << " ms\n";
      std::cout << "Update time: " << (update_end - update_start) / 1e6
                << " ms\n";
    }
  });

  // Set the timer interval to get 60 FPS
  frameTimer->start(1000 / 60);

  // Enter Qt event loop
  int ret = app.exec();

  // Clean up
  frameTimer->stop();

  return 0;
}
