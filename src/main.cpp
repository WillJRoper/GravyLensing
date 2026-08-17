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
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>

// Qt includes
#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QLibraryInfo>
#include <QMetaObject>
#include <QMetaType>
#include <QMessageBox>
#include <QSettings>
#include <QThread>
#include <QTimer>

// External includes
#include <fftw3.h>
#include <opencv2/opencv.hpp>

// Local includes
#include "backgrounds.hpp"
#include "cam_feed.hpp"
#include "cmd_parser.hpp"
#include "color_mask.hpp"
#include "lensing_worker.hpp"
#include "session_setup_dialog.hpp"
#include "segmentation_worker.hpp"
#include "settings.hpp"
#include "settings_dialog.hpp"
#include "viewport.hpp"

#ifdef __APPLE__
#include "apple_video_frame.hpp"
#endif

// Register cv::Mat as a Qt metatype
Q_DECLARE_METATYPE(cv::Mat)
#ifdef __APPLE__
Q_DECLARE_METATYPE(AppleVideoFrame)
#endif

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
 * @brief Connect the parts of the pipeline shared by all mask modes.
 */
void connectCommonSignals(CameraFeed *camFeed, LensingWorker *lensWorker,
                          ViewPort *vp, Backgrounds *backgrounds) {

  QObject::connect(lensWorker, &LensingWorker::lensedReady, vp,
                   &ViewPort::setLens, Qt::QueuedConnection);

  QObject::connect(backgrounds, &Backgrounds::backgroundChanged, lensWorker,
                   &LensingWorker::onBackgroundChange, Qt::QueuedConnection);

  QObject::connect(camFeed, &CameraFeed::frameCaptured, vp,
                   &ViewPort::setImage, Qt::QueuedConnection);
  QObject::connect(backgrounds, &Backgrounds::backgroundChanged, vp,
                   &ViewPort::setBackground, Qt::QueuedConnection);

  QObject::connect(camFeed, &CameraFeed::captureError, reportError);
  QObject::connect(lensWorker, &LensingWorker::lensingError, reportError);
}

/*
 * @brief Main function for the GravyLensing application.
 *
 * The application follows an explicit session lifecycle:
 *
 *   1. Load saved settings from QSettings.
 *   2. Parse CLI options, which override saved settings.
 *   3. Present the Session Setup dialog (user confirms configuration).
 *   4. Build and start the processing pipeline.
 *   5. Run the Qt event loop.
 *
 * Settings can be changed mid-session via Session > Session Settings...,
 * which tears down the current pipeline and restarts it with the new
 * configuration.  Live UI changes (mask mode, debug grid) update the
 * persistent settings immediately so they survive restarts.
 *
 * Pipeline stages:
 *   Capture (cam_feed) -> Mask (seg_worker or color_worker)
 *                       -> Lensing (lens_worker) -> UI (ViewPort)
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return int The exit code of the application.
 */
int main(int argc, char **argv) {

  enum class ActiveMaskMode { Person, Color };

  /// Transient session state that is persisted alongside AppSettings so that
  /// colour targets and ROI selections survive pipeline restarts.
  struct SessionSelections {
    bool hasColorTarget = false;
    float hue = 0.0f;
    float sat = 0.0f;
    float val = 0.0f;
    int hueTol = 0;
    int satTol = 0;
    int valTol = 0;
    bool hasROI = false;
    cv::Rect roiRect;
    cv::Mat roiMask;
  };

  // Homebrew's Qt packaging can place platform plugins under qtbase rather
  // than the more generic plugin path returned at runtime, so probe a few
  // likely roots and only accept one that contains the Cocoa platform plugin.
  const QStringList pluginRoots = {
      QLibraryInfo::path(QLibraryInfo::PluginsPath),
      "/opt/homebrew/opt/qtbase/share/qt/plugins",
      "/opt/homebrew/share/qt/plugins",
  };

  for (const QString &pluginPath : pluginRoots) {
    if (pluginPath.isEmpty())
      continue;

    const QString platformPath = pluginPath + "/platforms";
    const QString cocoaPlugin = platformPath + "/libqcocoa.dylib";
    if (!QFileInfo::exists(cocoaPlugin))
      continue;

    qputenv("QT_PLUGIN_PATH", pluginPath.toUtf8());
    qputenv("QT_QPA_PLATFORM_PLUGIN_PATH", platformPath.toUtf8());
    break;
  }

  // Set organisation / app name so QSettings stores in a predictable location
  QCoreApplication::setOrganizationName("GravyLensing");
  QCoreApplication::setApplicationName("gravy_lens");

  QApplication app(argc, argv);

  qRegisterMetaType<cv::Mat>("cv::Mat");
  qRegisterMetaType<cv::Rect>("cv::Rect");
#ifdef __APPLE__
  qRegisterMetaType<AppleVideoFrame>("AppleVideoFrame");
#endif

  QSettings savedSettings;
  AppSettings appSettings;
  appSettings.load(savedSettings);
  const int settingsVersion = savedSettings.value("settingsVersion", 0).toInt();
  if (settingsVersion < 1) {
    appSettings.flip = false;
    savedSettings.setValue("flip", false);
    savedSettings.setValue("settingsVersion", 1);
  }
#ifdef __APPLE__
  if (settingsVersion < 2 && appSettings.backgroundsDir == "backgrounds/") {
    appSettings.backgroundsDir = AppSettings().backgroundsDir;
    savedSettings.setValue(
        "backgroundsDir", QString::fromStdString(appSettings.backgroundsDir));
    savedSettings.setValue("settingsVersion", 2);
  }
#endif

  CommandLineOptions opts = CommandLineOptions::parse(app, appSettings);
  appSettings.nthreads = opts.nthreads;
  appSettings.strength = opts.strength;
  appSettings.softening = opts.softening;
  appSettings.deviceIndex = opts.deviceIndex;
  appSettings.fps = opts.fps;
  appSettings.debugGrid = opts.debugGrid;
  appSettings.padFactor = opts.padFactor;
  appSettings.visionSize = opts.visionSize;
  appSettings.temporalSmooth = opts.temporalSmooth;
  appSettings.lowerRes = opts.lowerRes;
  appSettings.personSensitivity = opts.personSensitivity;
  appSettings.qualityMode = opts.qualityMode;
  appSettings.secondsPerBackground = opts.secondsPerBackground;
  appSettings.distortInside = opts.distortInside;
  appSettings.flip = opts.flip;
  appSettings.selectROI = opts.selectROI;
  appSettings.maskMode = opts.maskMode;
  appSettings.colorModeType = opts.colorModeType;

  SessionSelections sessionSelections;
  const auto loadSessionSelections = [&](QSettings &settings) {
    sessionSelections.hasColorTarget =
        settings.value("session/hasColorTarget", false).toBool();
    sessionSelections.hue = settings.value("session/hue", 0.0).toFloat();
    sessionSelections.sat = settings.value("session/sat", 0.0).toFloat();
    sessionSelections.val = settings.value("session/val", 0.0).toFloat();
    sessionSelections.hueTol = settings.value("session/hueTol", 0).toInt();
    sessionSelections.satTol = settings.value("session/satTol", 0).toInt();
    sessionSelections.valTol = settings.value("session/valTol", 0).toInt();
    // ROI is session-only, not persisted across launches.
    sessionSelections.hasROI = false;
  };
  loadSessionSelections(savedSettings);

  {
    SessionSetupDialog startupDialog(
        appSettings, sessionSelections.hue, sessionSelections.sat,
        sessionSelections.val, sessionSelections.hasColorTarget,
        sessionSelections.hasROI);
    if (startupDialog.exec() != QDialog::Accepted) {
      return 0;
    }
    appSettings = startupDialog.settings();
    if (startupDialog.colorPickRequested()) {
      sessionSelections.hasColorTarget = true;
      sessionSelections.hue = startupDialog.pickedHue();
      sessionSelections.sat = startupDialog.pickedSat();
      sessionSelections.val = startupDialog.pickedVal();
    }
    if (startupDialog.colorFramePickRequested())
      sessionSelections.hasColorTarget = false;
  }

  fftwf_init_threads();

  Backgrounds *backgrounds =
      initBackgrounds(appSettings.backgroundsDir,
                      appSettings.backgroundWidth,
                      appSettings.backgroundHeight,
                      appSettings.backgroundFitMode,
                      appSettings.rebuildBackgroundCache);
  appSettings.rebuildBackgroundCache = false;
  ViewPort *vp = initViewport(backgrounds, appSettings, appSettings.debugGrid);

  AppSettings activeSettings = appSettings;
  CameraFeed *camFeed = nullptr;
  SegmentationWorker *segWorker = nullptr;
  ColorMaskWorker *colorWorker = nullptr;
  LensingWorker *lensWorker = nullptr;
  QThread *camThread = nullptr;
  QThread *maskThread = nullptr;
  QThread *lensThread = nullptr;
  QTimer *bgTimer = nullptr;
  QMetaObject::Connection frameToSegConnection;
  QMetaObject::Connection frameToColorConnection;
  bool personModeAvailable = false;
  ActiveMaskMode activeMaskMode = ActiveMaskMode::Person;
  std::function<void(ActiveMaskMode)> setActiveMaskMode;
  std::function<void()> updatePreviewPolicy;
  int renderedFrames = 0;
  int performanceWindows = 0;
  int lowPerformanceWindows = 0;
  bool performanceWarningShown = false;

  auto *performanceTimer = new QTimer(vp);
  performanceTimer->setInterval(5000);
  QObject::connect(performanceTimer, &QTimer::timeout, vp, [&]() {
    if (lensWorker == nullptr || camFeed == nullptr ||
        QApplication::applicationState() != Qt::ApplicationActive ||
        QApplication::activeModalWidget() != nullptr) {
      renderedFrames = 0;
      lowPerformanceWindows = 0;
      return;
    }
    const double actualFps = renderedFrames / 5.0;
    const double expectedFps =
        std::min(static_cast<double>(activeSettings.fps), camFeed->actualFps());
    renderedFrames = 0;
    if (++performanceWindows <= 2 || performanceWarningShown)
      return;

    if (expectedFps > 0.0 && actualFps < expectedFps * 0.55)
      ++lowPerformanceWindows;
    else
      lowPerformanceWindows = 0;

    if (lowPerformanceWindows < 3)
      return;

    performanceWarningShown = true;
    const AppSettings effective = activeSettings.withQualityModeApplied();
    QMessageBox::warning(
        vp, "Performance Below Target",
        QString("The effect is averaging about %1 fps, below the camera's %2 "
                "fps. Background output is %3 x %4 and internal calculation "
                "size is about %5 x %6.\n\nTry Fast quality first. You can also "
                "lower background resolution or camera frame rate.")
            .arg(actualFps, 0, 'f', 1)
            .arg(expectedFps, 0, 'f', 1)
            .arg(activeSettings.backgroundWidth)
            .arg(activeSettings.backgroundHeight)
            .arg(static_cast<int>(activeSettings.backgroundWidth *
                                  effective.lowerRes))
            .arg(static_cast<int>(activeSettings.backgroundHeight *
                                  effective.lowerRes)));
  });
  performanceTimer->start();

  const auto saveSessionSelections = [&](QSettings &settings) {
    settings.setValue("session/hasColorTarget", sessionSelections.hasColorTarget);
    settings.setValue("session/hue", sessionSelections.hue);
    settings.setValue("session/sat", sessionSelections.sat);
    settings.setValue("session/val", sessionSelections.val);
    settings.setValue("session/hueTol", sessionSelections.hueTol);
    settings.setValue("session/satTol", sessionSelections.satTol);
    settings.setValue("session/valTol", sessionSelections.valTol);
    // ROI is session-only; never persisted.
  };

  // ── Pipeline lifecycle helpers ─────────────────────────────────────
  // stopPipeline  – gracefully tears down all workers and threads.
  // startPipeline – builds and wires the full processing pipeline from
  //                 a settings snapshot and current session selections.

  const auto stopPipeline = [&]() {
    if (bgTimer != nullptr) {
      bgTimer->stop();
      delete bgTimer;
      bgTimer = nullptr;
    }

    // ── Shutdown ordering contract ─────────────────────────────────
    // The sequence below must be preserved:
    //
    // 1. stopCaptureLoop  – sets stopRequested_; the capture loop polls
    //    this and exits on its own.  It returns without waiting, so the
    //    camera thread may still be mid-emission for one last frame.
    //    This is safe because frame→worker connections are DirectConnection
    //    and the remaining steps gate work before teardown continues.
    // 2. disconnect signals – prevents any *future* camera-thread
    //    invocations of submitAppleFrame reaching the
    //    segmentation worker after this point.
    // 3. beginShutdown (BlockingQueuedConnection) – blocks the main
    //    thread until the mask thread has set shuttingDown_ and cleared
    //    all pending frame queues.  No queued drain should ever emit
    //    maskReady after this returns.
    // 4. deleteLater + thread quit/wait – defers object deletion to the
    //    owner thread, then drains the event loop so those deletions
    //    complete before the main thread touches any worker pointer.
    //
    // If BlockingQueuedConnection is ever relaxed to QueuedConnection,
    // deleteLater could race with in-flight drain events.
    if (camFeed != nullptr) {
      camFeed->stopCaptureLoop();
    }

    if (frameToSegConnection)
      QObject::disconnect(frameToSegConnection);
    if (frameToColorConnection)
      QObject::disconnect(frameToColorConnection);

    if (segWorker != nullptr) {
      QMetaObject::invokeMethod(segWorker, &SegmentationWorker::beginShutdown,
                                Qt::BlockingQueuedConnection);
    }

    // Post deferred-delete events to the workers' threads *before* we quit
    // those threads.  Otherwise the delete events can never be dispatched
    // and destruction on the main thread may touch stale thread-affinity
    // data, causing a crash.
    if (segWorker != nullptr)
      segWorker->deleteLater();
    if (colorWorker != nullptr)
      colorWorker->deleteLater();
    if (lensWorker != nullptr)
      lensWorker->deleteLater();
    if (camFeed != nullptr)
      camFeed->deleteLater();

    if (maskThread != nullptr) {
      maskThread->quit();
      maskThread->wait();
    }
    if (lensThread != nullptr) {
      lensThread->quit();
      lensThread->wait();
    }
    if (camThread != nullptr) {
      camThread->quit();
      camThread->wait();
    }

    delete maskThread;
    delete lensThread;
    delete camThread;

    camFeed = nullptr;
    segWorker = nullptr;
    colorWorker = nullptr;
    lensWorker = nullptr;
    maskThread = nullptr;
    lensThread = nullptr;
    camThread = nullptr;
    personModeAvailable = false;
    frameToSegConnection = QMetaObject::Connection();
    frameToColorConnection = QMetaObject::Connection();
    updatePreviewPolicy = nullptr;
  };

  const auto attachPersonWorker = [&](SegmentationWorker *worker) {
    if (worker == nullptr) {
      return;
    }

    QObject::connect(worker, &SegmentationWorker::maskReady, lensWorker,
                     [lensWorker](const cv::Mat &mask, quint64 seq) {
                       if (lensWorker)
                         lensWorker->submitMask(mask, seq);
                     },
                     Qt::DirectConnection);
    QObject::connect(backgrounds, &Backgrounds::backgroundChanged, worker,
                     &SegmentationWorker::onBackgroundChange,
                     Qt::QueuedConnection);
    QObject::connect(worker, &SegmentationWorker::segmentationError,
                     reportError);
    QObject::connect(worker, &SegmentationWorker::maskReady, vp,
                     &ViewPort::setMask, Qt::QueuedConnection);
  };

  const auto ensurePersonWorkerLoaded = [&]() -> bool {
    if (segWorker != nullptr) {
      return personModeAvailable;
    }

    const AppSettings effectiveSettings = activeSettings.withQualityModeApplied();
    SegmentationWorker *newSegWorker =
        new SegmentationWorker(effectiveSettings.visionSize,
                               effectiveSettings.temporalSmooth,
                               effectiveSettings.lowerRes,
                               effectiveSettings.visionQualityMode(),
                               effectiveSettings.personSensitivity);
    if (!newSegWorker->isReady()) {
      reportError("Apple Vision person segmentation is unavailable");
      delete newSegWorker;
      return false;
    }

    newSegWorker->moveToThread(maskThread);
    attachPersonWorker(newSegWorker);
    segWorker = newSegWorker;
    personModeAvailable = true;

    QMetaObject::invokeMethod(segWorker,
                              [worker = segWorker]() { worker->setEnabled(false); },
                              Qt::QueuedConnection);
    emit backgrounds->backgroundChanged(backgrounds->current());
    return true;
  };

  const auto startPipeline = [&](const AppSettings &settings,
                                 const SessionSelections &selections,
                                 bool isReconfigure = false) -> bool {
    const AppSettings effectiveSettings = settings.withQualityModeApplied();
    renderedFrames = 0;
    performanceWindows = 0;
    lowPerformanceWindows = 0;
    performanceWarningShown = false;
    fftwf_plan_with_nthreads(settings.nthreads);

    // On a reconfigure we never auto-open the blocking ROI selector,
    // even when the saved setting says selectROI is true.
    const bool showROI = settings.selectROI && !isReconfigure;

    CameraFeed *newCamFeed = new CameraFeed(settings.deviceIndex, settings.flip,
                                             showROI, settings.fps,
                                             settings.cameraWidth,
                                             settings.cameraHeight);
    if (!newCamFeed->isOpen()) {
      delete newCamFeed;
      return false;
    }
    vp->setWindowTitle(
        QString("GravyLensing - Camera %1 x %2 at %3 fps")
            .arg(newCamFeed->actualWidth())
            .arg(newCamFeed->actualHeight())
            .arg(newCamFeed->actualFps(), 0, 'f', 1));

    SegmentationWorker *newSegWorker = nullptr;
    bool newPersonModeAvailable = false;
    if (settings.maskMode == "person") {
      newSegWorker = new SegmentationWorker(
          effectiveSettings.visionSize, effectiveSettings.temporalSmooth,
          effectiveSettings.lowerRes, effectiveSettings.visionQualityMode(),
          effectiveSettings.personSensitivity);
      newPersonModeAvailable = newSegWorker->isReady();
      if (!newPersonModeAvailable) {
        reportError("Apple Vision person segmentation is unavailable");
        delete newSegWorker;
        delete newCamFeed;
        return false;
      }
    }

    // Colour mode is always started without an automatic picker.  The user
    // explicitly chooses/re-chooses the target via menu or key binding.
    ColorMaskWorker *newColorWorker =
        new ColorMaskWorker(effectiveSettings.lowerRes);

    if (selections.hasColorTarget) {
      newColorWorker->applyReselectionTarget(selections.hue, selections.sat,
                                             selections.val, selections.hueTol,
                                             selections.satTol,
                                             selections.valTol, true);
    }
    newColorWorker->setTrackedBlobMode(settings.colorModeType == "tracked_blob");
    newColorWorker->setTolerances(settings.colorHueTol, settings.colorSatTol,
                                  settings.colorValTol);
    newColorWorker->setTrackingTuning(
        settings.colorMinObjectArea, settings.colorPersistenceFrames,
        settings.colorMaskSmooth);

    LensingWorker *newLensWorker =
        new LensingWorker(settings.strength, settings.softening,
                           settings.padFactor, settings.nthreads,
                           effectiveSettings.lowerRes, settings.distortInside,
                           effectiveSettings.lensMassBlurSigma());
    QObject::connect(newLensWorker, &LensingWorker::lensedReady, vp,
                     [&renderedFrames](const cv::Mat &) { ++renderedFrames; },
                     Qt::QueuedConnection);

    QThread *newMaskThread = new QThread;
    QThread *newLensThread = new QThread;
    QThread *newCamThread = new QThread;

    // Apply saved ROI *before* moving the camera feed to its thread so that
    // all direct method calls happen on the calling (main) thread.
    if (selections.hasROI && !showROI) {
      cv::Mat roiMask = selections.roiMask.clone();
      if (roiMask.empty() && selections.roiRect.width > 0 &&
          selections.roiRect.height > 0) {
        const cv::Mat selectionFrame = newCamFeed->captureSelectionFrame();
        if (!selectionFrame.empty() && selections.roiRect.x >= 0 &&
            selections.roiRect.y >= 0 &&
            selections.roiRect.x + selections.roiRect.width <=
                selectionFrame.cols &&
            selections.roiRect.y + selections.roiRect.height <=
                selectionFrame.rows) {
          roiMask = cv::Mat(selectionFrame.size(), CV_8UC1, cv::Scalar(0));
          cv::rectangle(roiMask, selections.roiRect, cv::Scalar(255),
                        cv::FILLED);
        }
      }

      if (!roiMask.empty()) {
        newCamFeed->setROI(selections.roiRect, roiMask);
      }
    }

    if (newSegWorker != nullptr) {
      newSegWorker->moveToThread(newMaskThread);
    }
    newColorWorker->moveToThread(newMaskThread);
    newLensWorker->moveToThread(newLensThread);
    newCamFeed->moveToThread(newCamThread);

    QObject::connect(newCamThread, &QThread::started, newCamFeed,
                     &CameraFeed::startCaptureLoop);

    connectCommonSignals(newCamFeed, newLensWorker, vp, backgrounds);

    if (newSegWorker != nullptr) {
      QObject::connect(newSegWorker, &SegmentationWorker::maskReady,
                       newLensWorker,
                       [newLensWorker](const cv::Mat &mask, quint64 seq) {
                         newLensWorker->submitMask(mask, seq);
                       },
                       Qt::DirectConnection);
      QObject::connect(backgrounds, &Backgrounds::backgroundChanged,
                       newSegWorker, &SegmentationWorker::onBackgroundChange,
                       Qt::QueuedConnection);
      QObject::connect(newSegWorker, &SegmentationWorker::segmentationError,
                       reportError);
    }

    QObject::connect(newColorWorker, &ColorMaskWorker::maskReady,
                     newLensWorker, [newLensWorker](const cv::Mat &mask) {
                       newLensWorker->submitMask(mask);
                     },
                     Qt::DirectConnection);
    QObject::connect(backgrounds, &Backgrounds::backgroundChanged,
                     newColorWorker, &ColorMaskWorker::onBackgroundChange,
                     Qt::QueuedConnection);
    QObject::connect(newColorWorker, &ColorMaskWorker::maskReady, vp,
                     &ViewPort::setMask, Qt::QueuedConnection);
    if (newSegWorker != nullptr) {
      QObject::connect(newSegWorker, &SegmentationWorker::maskReady, vp,
                       &ViewPort::setMask, Qt::QueuedConnection);
    }
    QObject::connect(newColorWorker, &ColorMaskWorker::maskError, reportError);

    QObject::connect(newColorWorker,
                     &ColorMaskWorker::reselectionRequested, vp,
                     [&, colorWorker = newColorWorker](const cv::Mat &frame) {
                         const auto stats =
                             ColorMaskWorker::runInteractiveColorPicker(frame);
                         const bool success = stats.count > 0;
                        const float kSpreadScale = 2.5f;
                        const int hueTol = std::max(
                            12, static_cast<int>(std::ceil(stats.hueSpread * kSpreadScale)));
                        const int satTol = std::max(
                            120, static_cast<int>(std::ceil(stats.satSpread * kSpreadScale)));
                        const int valTol = std::max(
                            180, static_cast<int>(std::ceil(stats.valSpread * kSpreadScale)));
                        QMetaObject::invokeMethod(
                            colorWorker, "applyReselectionTarget",
                            Qt::QueuedConnection,
                            Q_ARG(float, stats.hue),
                            Q_ARG(float, stats.sat),
                            Q_ARG(float, stats.val),
                            Q_ARG(int, hueTol),
                             Q_ARG(int, satTol),
                             Q_ARG(int, valTol),
                             Q_ARG(bool, success));

                          if (success) {
                            sessionSelections.hasColorTarget = true;
                            sessionSelections.hue = stats.hue;
                            sessionSelections.sat = stats.sat;
                            sessionSelections.val = stats.val;
                            sessionSelections.hueTol = hueTol;
                            sessionSelections.satTol = satTol;
                            sessionSelections.valTol = valTol;
                            activeSettings.colorHueTol = hueTol;
                            activeSettings.colorSatTol = satTol;
                            activeSettings.colorValTol = valTol;
                            vp->setSettings(activeSettings);
                            QSettings s;
                            activeSettings.save(s);
                            saveSessionSelections(s);
                         }
                       },
                     Qt::QueuedConnection);

    QObject::connect(newColorWorker, &ColorMaskWorker::selectionStateChanged,
                     vp,
                     [&](bool ready) {
                       if (!ready && activeMaskMode == ActiveMaskMode::Color &&
                           personModeAvailable) {
                         std::cout << "[Main] Color selection cancelled; "
                                      "reverting to person mode\n";
                         if (setActiveMaskMode)
                           setActiveMaskMode(ActiveMaskMode::Person);
                       }
                     },
                     Qt::QueuedConnection);

    newMaskThread->start();
    newLensThread->start();

    camFeed = newCamFeed;
    segWorker = newSegWorker;
    colorWorker = newColorWorker;
    lensWorker = newLensWorker;
    maskThread = newMaskThread;
    lensThread = newLensThread;
    camThread = newCamThread;
    personModeAvailable = newPersonModeAvailable;

    updatePreviewPolicy = [&]() {
      if (camFeed == nullptr) {
        return;
      }
      const bool useNativeOnlyPersonPath =
          activeMaskMode == ActiveMaskMode::Person && segWorker != nullptr &&
          !activeSettings.debugGrid && !activeSettings.showLensContents;
      QMetaObject::invokeMethod(
          camFeed,
          [camFeed, enablePreview = !useNativeOnlyPersonPath]() {
            if (camFeed) {
              camFeed->setPreviewEnabled(enablePreview);
            }
          },
          Qt::QueuedConnection);
    };

    setActiveMaskMode = [&](ActiveMaskMode mode) {
      if (mode == ActiveMaskMode::Person && !ensurePersonWorkerLoaded()) {
        return;
      }
      activeMaskMode = mode;

      const bool enablePerson = mode == ActiveMaskMode::Person;
      const bool enableColor = mode == ActiveMaskMode::Color;

      if (frameToSegConnection)
        QObject::disconnect(frameToSegConnection);
      if (frameToColorConnection)
        QObject::disconnect(frameToColorConnection);

      if (enablePerson && segWorker != nullptr) {
        frameToSegConnection = QObject::connect(
            camFeed, &CameraFeed::framePairCaptured, segWorker,
            [segWorker](const cv::Mat &frame,
                        const AppleVideoFrame &nativeFrame, quint64 seq) {
              segWorker->submitAppleFrame(nativeFrame, frame, seq);
            },
            Qt::DirectConnection);
      }
      if (enableColor) {
        frameToColorConnection = QObject::connect(
            camFeed, &CameraFeed::frameCaptured, colorWorker,
            [colorWorker](const cv::Mat &frame) {
              colorWorker->submitFrame(frame);
            },
            Qt::DirectConnection);
      }

      if (segWorker != nullptr) {
        QMetaObject::invokeMethod(
            segWorker,
            [segWorker, enablePerson]() {
              if (segWorker)
                segWorker->setEnabled(enablePerson);
            },
            Qt::QueuedConnection);
      }
      QMetaObject::invokeMethod(
          colorWorker,
          [colorWorker, enableColor]() {
            if (colorWorker) colorWorker->setEnabled(enableColor);
          },
          Qt::QueuedConnection);

      vp->setColorModeActive(enableColor);
      vp->setMaskModeLabel(mode == ActiveMaskMode::Color);
      if (updatePreviewPolicy)
        updatePreviewPolicy();

      if (enableColor && !sessionSelections.hasColorTarget) {
        std::cout << "[Main] Color mode active with no selected target. "
                     "Use Shift+S or File > Select Color...\n";
      }

      std::cout << "[Main] Active mask mode: "
                << (mode == ActiveMaskMode::Person ? "person" : "color")
                << "\n";
    };

    vp->setSettings(settings);
    vp->setColorTarget(selections.hue, selections.sat, selections.val,
                        selections.hasColorTarget);
    vp->setROIState(selections.hasROI,
                     selections.roiRect.x, selections.roiRect.y,
                     selections.roiRect.width, selections.roiRect.height);
    vp->setDebugGridEnabled(settings.debugGrid);

    const ActiveMaskMode desiredMode =
        settings.maskMode == "color" ? ActiveMaskMode::Color
                                       : ActiveMaskMode::Person;
    setActiveMaskMode(desiredMode);
    vp->setDebugGridChecked(settings.debugGrid);

    vp->setBackground(backgrounds->current());
    emit backgrounds->backgroundChanged(backgrounds->current());
    newCamThread->start();  // camera last — workers primed first

    if (settings.secondsPerBackground > 0) {
      bgTimer = new QTimer(vp);
      QObject::connect(bgTimer, &QTimer::timeout, backgrounds,
                       &Backgrounds::next);
      bgTimer->start(settings.secondsPerBackground * 1000);
    }

    return true;
  };

  if (!startPipeline(activeSettings, sessionSelections)) {
    return -1;
  }

  activeSettings.save(savedSettings);
  saveSessionSelections(savedSettings);

  if (activeSettings.maskMode == "color" && !sessionSelections.hasColorTarget &&
      colorWorker != nullptr) {
    QMetaObject::invokeMethod(colorWorker, &ColorMaskWorker::triggerReselect,
                              Qt::QueuedConnection);
  }

  // ── ViewPort menu signals → pipeline actions ───────────────────────
  // These connections handle user-initiated events from the menu bar
  // during an active session.

  QObject::connect(vp, &ViewPort::debugGridToggled, vp,
                   [vp, &activeSettings, &saveSessionSelections,
                    &updatePreviewPolicy](bool enabled) {
                      vp->setDebugGridEnabled(enabled);
                      vp->setDebugGridChecked(enabled);
                      activeSettings.debugGrid = enabled;
                      vp->setSettings(activeSettings);
                      if (updatePreviewPolicy)
                        updatePreviewPolicy();
                      QSettings s;
                     activeSettings.save(s);
                     saveSessionSelections(s);
                    });

  QObject::connect(vp, &ViewPort::showLensContentsToggled, vp,
                   [vp, &activeSettings, &saveSessionSelections,
                    &updatePreviewPolicy](bool enabled) {
                     activeSettings.showLensContents = enabled;
                     vp->setSettings(activeSettings);
                     if (updatePreviewPolicy)
                       updatePreviewPolicy();
                     QSettings s;
                     activeSettings.save(s);
                     saveSessionSelections(s);
                   });

  QObject::connect(vp, &ViewPort::selectROIRequested, vp, [&]() {
    if (camFeed == nullptr || camThread == nullptr) {
      return;
    }

    camFeed->stopCaptureLoop();
    camThread->quit();
    camThread->wait();

    cv::Mat frame = camFeed->captureSelectionFrame();
    if (!frame.empty()) {
      const auto [rect, mask] = selectROIAndMask(frame, false);
      if (rect.width > 0 && rect.height > 0) {
        camFeed->setROI(rect, mask);
        sessionSelections.hasROI = true;
        sessionSelections.roiRect = rect;
        sessionSelections.roiMask = mask.clone();
        vp->setROIState(true, rect.x, rect.y, rect.width, rect.height);
        QSettings s;
        activeSettings.save(s);
        saveSessionSelections(s);
      }
    }

    camThread->start();
  });

  QObject::connect(vp, &ViewPort::clearROIRequested, vp, [&]() {
    sessionSelections.hasROI = false;
    sessionSelections.roiRect = cv::Rect();
    sessionSelections.roiMask.release();
    if (camFeed != nullptr)
      camFeed->clearROI();
    vp->setROIState(false, 0, 0, 0, 0);
  });

  QObject::connect(vp, &ViewPort::selectColorRequested, vp, [&]() {
    if (colorWorker == nullptr || !setActiveMaskMode)
      return;

    if (activeMaskMode != ActiveMaskMode::Color) {
      setActiveMaskMode(ActiveMaskMode::Color);
    }

    QMetaObject::invokeMethod(colorWorker, &ColorMaskWorker::triggerReselect,
                              Qt::QueuedConnection);
  });

  const auto requestMaskMode = [&](ActiveMaskMode requestedMode) {
    if (!colorWorker || !setActiveMaskMode || activeMaskMode == requestedMode)
      return;

    setActiveMaskMode(requestedMode);
    if (activeMaskMode != requestedMode)
      return;
    activeSettings.maskMode = requestedMode == ActiveMaskMode::Color
                                  ? "color"
                                  : "person";
    vp->setSettings(activeSettings);
    QSettings s;
    activeSettings.save(s);
    saveSessionSelections(s);
  };

  QObject::connect(vp, &ViewPort::toggleMaskModeRequested, vp, [&]() {
    requestMaskMode(activeMaskMode == ActiveMaskMode::Color
                        ? ActiveMaskMode::Person
                        : ActiveMaskMode::Color);
  });
  QObject::connect(vp, &ViewPort::maskModeRequested, vp,
                   [&](bool colorMode) {
                     requestMaskMode(colorMode ? ActiveMaskMode::Color
                                               : ActiveMaskMode::Person);
                   });

  int lastBackgroundInterval =
      activeSettings.secondsPerBackground > 0
          ? activeSettings.secondsPerBackground
          : 10;
  const auto setBackgroundCycleInterval = [&](int seconds) {
    if (bgTimer != nullptr) {
      bgTimer->stop();
      delete bgTimer;
      bgTimer = nullptr;
    }
    activeSettings.secondsPerBackground = seconds;
    if (seconds > 0) {
      lastBackgroundInterval = seconds;
      bgTimer = new QTimer(vp);
      QObject::connect(bgTimer, &QTimer::timeout, backgrounds,
                       &Backgrounds::next);
      bgTimer->start(seconds * 1000);
    }
    vp->setSettings(activeSettings);
    QSettings s;
    activeSettings.save(s);
    saveSessionSelections(s);
  };

  QObject::connect(vp, &ViewPort::backgroundAutoCycleToggled, vp,
                   [&](bool enabled) {
                     setBackgroundCycleInterval(enabled
                                                    ? lastBackgroundInterval
                                                    : -1);
                   });
  QObject::connect(vp, &ViewPort::backgroundIntervalRequested, vp,
                   [&](int seconds) {
                     setBackgroundCycleInterval(seconds);
                   });

  // ── Session restart handler ────────────────────────────────────────
  // Triggered by the Session Settings dialog.  Preserves the current
  // colour target and ROI, tears down the pipeline, and rebuilds it
  // with the new settings.  Falls back to the previous configuration
  // if the new one fails.

  QObject::connect(vp, &ViewPort::settingsChanged, vp,
                   [&](const AppSettings &newSettings) {
                      if (newSettings.equals(activeSettings)) {
                        vp->setSettings(activeSettings);
                        return;
                      }

                       const AppSettings previousSettings = activeSettings;
                       const SessionSelections previousSelections =
                           sessionSelections;
                        const bool backgroundsChanged =
                            newSettings.backgroundsDir !=
                                previousSettings.backgroundsDir ||
                            newSettings.backgroundWidth !=
                                previousSettings.backgroundWidth ||
                            newSettings.backgroundHeight !=
                                previousSettings.backgroundHeight ||
                            newSettings.backgroundFitMode !=
                                previousSettings.backgroundFitMode ||
                            newSettings.rebuildBackgroundCache;
                       const bool actuallySwitchedToColor =
                           previousSettings.maskMode != "color" &&
                           newSettings.maskMode == "color";

                       if (vp->hasColorTarget()) {
                         sessionSelections.hasColorTarget = true;
                         sessionSelections.hue = vp->targetHue();
                         sessionSelections.sat = vp->targetSat();
                         sessionSelections.val = vp->targetVal();
                         sessionSelections.hueTol = newSettings.colorHueTol;
                         sessionSelections.satTol = newSettings.colorSatTol;
                         sessionSelections.valTol = newSettings.colorValTol;
                       } else {
                         sessionSelections.hasColorTarget = false;
                         sessionSelections.hue = 0.0f;
                         sessionSelections.sat = 0.0f;
                         sessionSelections.val = 0.0f;
                       }

                       // sessionSelections is already up-to-date from the
                       // live handlers (reselection lambda, ROI handler).
                      // No need to BlockingQueuedConnection-query workers.

                       if (backgroundsChanged &&
                           !backgrounds->setSource(
                               newSettings.backgroundsDir,
                               newSettings.backgroundWidth,
                               newSettings.backgroundHeight,
                               newSettings.backgroundFitMode,
                               newSettings.rebuildBackgroundCache)) {
                         QMessageBox::warning(
                             vp, "Backgrounds Not Changed",
                             "No supported images could be loaded from the "
                             "selected directory.");
                         vp->setSettings(previousSettings);
                         return;
                       }
                       if (backgroundsChanged)
                         vp->setBackgroundImages(backgrounds);

                       stopPipeline();

                      if (!startPipeline(newSettings, sessionSelections,
                                         /*isReconfigure=*/true)) {
                        QMessageBox::warning(
                            vp, "Settings Not Applied",
                            "The new settings could not be applied. Restoring "
                            "the previous working configuration.");

                         sessionSelections = previousSelections;

                         if (backgroundsChanged) {
                           backgrounds->setSource(
                               previousSettings.backgroundsDir,
                               previousSettings.backgroundWidth,
                               previousSettings.backgroundHeight,
                               previousSettings.backgroundFitMode);
                           vp->setBackgroundImages(backgrounds);
                         }

                         if (!startPipeline(previousSettings, previousSelections,
                                           true)) {
                          QMessageBox::critical(
                              vp, "Fatal Configuration Error",
                              "The app could not restore the previous working "
                             "configuration. It will now exit.");
                         qApp->quit();
                         return;
                       }

                       vp->setSettings(previousSettings);
                       activeSettings = previousSettings;
                       return;
                     }

                       activeSettings = newSettings;
                       activeSettings.rebuildBackgroundCache = false;
                       vp->setSettings(activeSettings);
                      vp->setColorTarget(sessionSelections.hue,
                                          sessionSelections.sat,
                                          sessionSelections.val,
                                          sessionSelections.hasColorTarget);
                      QSettings s;
                      activeSettings.save(s);
                      saveSessionSelections(s);

                      // Only auto-trigger the colour picker when the user
                      // deliberately switches *into* colour mode.  If we are
                      // already in colour mode we keep the current target.
                      if (actuallySwitchedToColor &&
                          !sessionSelections.hasColorTarget) {
                        std::cout << "[Main] Switched to color mode. Use "
                                     "Shift+S or File > Select Color... to "
                                     "choose a target.\n";
                      }
                   });

  int ret = app.exec();

  stopPipeline();
  return ret;
}
