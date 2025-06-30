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
#include <chrono>
#include <iostream>
#include <string>

// Qt includes
#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QMetaType>
#include <QThread>
#include <QTimer>

// External includes
#include <fftw3.h>
#include <opencv2/opencv.hpp>

// Local includes
#include "backgrounds.hpp"
#include "cam_feed.hpp"
#include "cmd_parser.hpp"
#include "lensing_worker.hpp"
#include "segmentation_worker.hpp"
#include "viewport.hpp"

// Define the path to the background images (this is constant)
const std::string bgDir = "backgrounds/";

// Register cv::Mat as a Qt metatype
Q_DECLARE_METATYPE(cv::Mat)

/**
 * @brief Report any errors that occur during the application execution.
 *
 * This function is used to report errors that occur during the execution
 * of the GravyLensing application. It can be used to log errors or display
 * them to the user.
 *
 * @param err The error message to report.
 */
void reportError(const std::string &err) {
  std::cerr << "Error: " << err << std::endl;
}

/**
 * @brief Connected all the signals and slots.
 *
 * This function connects the signals and slots between the camera feed,
 * segmentation worker, lensing worker, and the viewport.
 *
 * @param camFeed The camera feed object.
 * @param segWorker The segmentation worker object.
 * @param lensWorker The lensing worker object.
 * @param vp The viewport object.
 * @param backgrounds The backgrounds object.
 * @param debugGrid Whether to enable the debug grid.
 */
void connectSignals(CameraFeed *camFeed, SegmentationWorker *segWorker,
                    LensingWorker *lensWorker, ViewPort *vp,
                    Backgrounds *backgrounds, bool debugGrid) {

  // First the main steps of the calculation:
  //      Frame -> Segmentation - > Lensing - > ViewPort

  // Camera → Segmentation (New frame)
  QObject::connect(camFeed, &CameraFeed::frameCaptured, segWorker,
                   &SegmentationWorker::onFrame, Qt::QueuedConnection);

  // Segmentation → Lensing (New mask)
  QObject::connect(segWorker, &SegmentationWorker::maskReady, lensWorker,
                   &LensingWorker::onMask, Qt::QueuedConnection);

  // ViewPort ← Lensing (New lensed image)
  QObject::connect(lensWorker, &LensingWorker::lensedReady, vp,
                   &ViewPort::setLens, Qt::QueuedConnection);

  // Next connect up and the things that run when the background changes:

  // Backgrounds → Segmentation (New background)
  QObject::connect(backgrounds, &Backgrounds::backgroundChanged, segWorker,
                   &SegmentationWorker::onBackgroundChange,
                   Qt::QueuedConnection);

  // Background switches from UI → LensingWorker
  QObject::connect(backgrounds, &Backgrounds::backgroundChanged, lensWorker,
                   &LensingWorker::onBackgroundChange, Qt::QueuedConnection);

  // Handle the debug grid specific connections for extra displays in the
  // viewport
  if (debugGrid) {

    // ViewPort ← Camera (raw display)
    QObject::connect(camFeed, &CameraFeed::frameCaptured, vp,
                     &ViewPort::setImage, Qt::QueuedConnection);

    // ViewPort ← Segmentation (mask display, if in debug)
    QObject::connect(segWorker, &SegmentationWorker::maskReady, vp,
                     &ViewPort::setMask, Qt::QueuedConnection);

    // Also update the UI display when background changes
    QObject::connect(backgrounds, &Backgrounds::backgroundChanged, vp,
                     &ViewPort::setBackground, Qt::QueuedConnection);
  }

  // Link up error reporting
  QObject::connect(camFeed, &CameraFeed::captureError, reportError);
  QObject::connect(segWorker, &SegmentationWorker::segmentationError,
                   reportError);
  QObject::connect(lensWorker, &LensingWorker::lensingError, reportError);
}

/*
 * @brief Main function for the GravyLensing application.
 *
 * This function initializes the application, parses command-line options,
 * sets up the camera feed, segmentation, and lensing workers, and starts
 * the event loop.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return int The exit code of the application.
 */
int main(int argc, char **argv) {
  QApplication app(argc, argv);

  // Parse options
  CommandLineOptions opts = CommandLineOptions::parse(app);
  int nthreads = opts.nthreads;
  float strength = opts.strength;
  float softening = opts.softening;
  int padFactor = opts.padFactor;
  bool debugGrid = opts.debugGrid;
  int modelSize = opts.modelSize;
  int deviceIndex = opts.deviceIndex;
  float temporalSmooth = opts.temporalSmooth;
  float lowerRes = opts.lowerRes;
  int secondsPerBackground = opts.secondsPerBackground;
  bool distortInside = opts.distortInside;
  bool flip = opts.flip;
  bool selectROI = opts.selectROI;
  const std::string modelPath = opts.modelPath;

  // Correct the number of threads to account for those that have
  // been taken by Qt
  nthreads -= 3;

  // Init FFTW threading
  fftwf_init_threads();
  fftwf_plan_with_nthreads(nthreads);

  // Load backgrounds from the specified directory
  Backgrounds *backgrounds = initBackgrounds(bgDir);

  // Create UI
  ViewPort *vp = initViewport(backgrounds, debugGrid);

  // Create workers & threads

  // Camera feed
  CameraFeed *camFeed = new CameraFeed(deviceIndex, flip, selectROI);
  QThread *camThread = new QThread;
  camFeed->moveToThread(camThread);
  QObject::connect(camThread, &QThread::started, camFeed,
                   &CameraFeed::startCaptureLoop);
  camThread->start();

  // Segmentation
  auto segWorker = new SegmentationWorker(modelPath, modelSize, nthreads,
                                          temporalSmooth, lowerRes);
  QThread *segThread = new QThread;
  segWorker->moveToThread(segThread);
  segThread->start();

  // Lensing
  auto lensWorker = new LensingWorker(strength, softening, padFactor, nthreads,
                                      lowerRes, distortInside);
  QThread *lensThread = new QThread;
  lensWorker->moveToThread(lensThread);
  lensThread->start();

  // Wire up signals/slots
  connectSignals(camFeed, segWorker, lensWorker, vp, backgrounds, debugGrid);

  // Prime initial background
  vp->setBackground(backgrounds->current());
  emit backgrounds->backgroundChanged(backgrounds->current());

  // If we are updating each background every X seconds, set up a timer to do
  // that. IF not secondsPerBackground is -1 and we don't do anything.
  if (secondsPerBackground > 0) {
    QTimer *timer = new QTimer();
    QObject::connect(timer, &QTimer::timeout, backgrounds, &Backgrounds::next);
    timer->start(secondsPerBackground * 1000);
  }

  // Run!
  int ret = app.exec();

  // 9) Cleanup threads
  segThread->quit();
  segThread->wait();
  lensThread->quit();
  lensThread->wait();
  camThread->quit();
  camThread->wait();

  delete camFeed;
  delete segWorker;
  delete lensWorker;
  delete camThread;
  delete segThread;
  delete lensThread;

  return ret;
}
