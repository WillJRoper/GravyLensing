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

// Qt includes
#include <QApplication>
#include <QCoreApplication>
#include <QMetaType>
#include <QThread>
#include <QTimer>

// External includes
#include <fftw3.h>
#include <opencv2/opencv.hpp>

// Local includes
#include "VisionSegmentationWorker.hpp"
#include "backgrounds.hpp"
#include "cam_feed.hpp"
#include "cmd_parser.hpp"
#include "lensing_worker.hpp"
#include "viewport.hpp"

// Path to background images
const std::string bgDir = "backgrounds/";

// Register cv::Mat as a Qt metatype
Q_DECLARE_METATYPE(cv::Mat)

/**
 * @brief Connect the static parts of the pipeline (lensing and UI).
 */
void connectCoreSignals(CameraFeed *camFeed, LensingWorker *lensWorker,
                        ViewPort *vp, Backgrounds *backgrounds,
                        bool debugGrid) {
  // Lensing → ViewPort
  QObject::connect(lensWorker, &LensingWorker::lensedReady, vp,
                   &ViewPort::setLens, Qt::QueuedConnection);

  // Background → Lensing
  QObject::connect(backgrounds, &Backgrounds::backgroundChanged, lensWorker,
                   &LensingWorker::onBackgroundChange, Qt::QueuedConnection);

  if (debugGrid) {
    // Raw camera preview
    QObject::connect(camFeed, &CameraFeed::frameCaptured, vp,
                     &ViewPort::setImage, Qt::QueuedConnection);
    // Show background in debug UI
    QObject::connect(backgrounds, &Backgrounds::backgroundChanged, vp,
                     &ViewPort::setBackground, Qt::QueuedConnection);
  }
}

/**
 * @brief Main function.
 */
int main(int argc, char **argv) {
  QApplication app(argc, argv);

  // Parse CLI options
  CommandLineOptions opts = CommandLineOptions::parse(app);
  int nthreads = opts.nthreads - 3; // reserve for Qt
  float strength = opts.strength;
  float softening = opts.softening;
  int padFactor = opts.padFactor;
  bool debugGrid = opts.debugGrid;
  int modelSize = opts.modelSize; // unused by Vision
  int deviceIndex = opts.deviceIndex;
  float temporalSmooth = opts.temporalSmooth; // unused by Vision
  float lowerRes = opts.lowerRes;             // unused by Vision
  int secondsPerBackground = opts.secondsPerBackground;
  bool distortInside = opts.distortInside;
  bool flip = opts.flip;
  bool selectROI = opts.selectROI;
  const std::string modelPath = opts.modelPath; // unused by Vision

  // Init FFTW threading
  fftwf_init_threads();
  fftwf_plan_with_nthreads(nthreads);

  // Load backgrounds
  Backgrounds *backgrounds = initBackgrounds(bgDir);

  // Create UI
  ViewPort *vp = initViewport(backgrounds, debugGrid);

  // Camera feed
  CameraFeed *camFeed = new CameraFeed(deviceIndex, flip, selectROI);
  QThread *camThread = new QThread;
  camFeed->moveToThread(camThread);
  QObject::connect(camThread, &QThread::started, camFeed,
                   &CameraFeed::startCaptureLoop);
  QObject::connect(camFeed, &CameraFeed::captureError, vp,
                   [&](const QString &err) { qWarning() << err; });
  camThread->start();

  // Vision‐based segmentation
  VisionSegmentationWorker *segWorker = new VisionSegmentationWorker;
  QThread *segThread = new QThread;
  segWorker->moveToThread(segThread);
  segThread->start();

  // Lensing worker
  LensingWorker *lensWorker = new LensingWorker(
      strength, softening, padFactor, nthreads, lowerRes, distortInside);
  QThread *lensThread = new QThread;
  lensWorker->moveToThread(lensThread);
  lensThread->start();

  // Core signal wiring
  connectCoreSignals(camFeed, lensWorker, vp, backgrounds, debugGrid);

  // Camera → Segmentation
  QObject::connect(camFeed, &CameraFeed::frameCaptured, segWorker,
                   &VisionSegmentationWorker::onFrame, Qt::QueuedConnection);

  // Segmentation → Lensing
  QObject::connect(segWorker, &VisionSegmentationWorker::maskReady, lensWorker,
                   &LensingWorker::onMask, Qt::QueuedConnection);

  // Prime initial background
  vp->setBackground(backgrounds->current());
  emit backgrounds->backgroundChanged(backgrounds->current());

  // Rotate backgrounds if desired
  if (secondsPerBackground > 0) {
    QTimer *timer = new QTimer;
    QObject::connect(timer, &QTimer::timeout, backgrounds, &Backgrounds::next);
    timer->start(secondsPerBackground * 1000);
  }

  // Start Qt event loop
  int ret = app.exec();

  // Cleanup
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
  delete backgrounds;
  delete vp;

  return ret;
}
