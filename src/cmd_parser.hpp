/**
 * @file cmd_parser.hpp
 *
 * Defines the interface to the command-line parser for GravyLensing.
 *
 * This class is responsible for parsing command-line options using Qt's
 * QCommandLineParser. It provides a convenient way to handle various
 * command-line arguments and validate them.
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

#pragma once

// Standard includes
#include <string>

// Qt includes
#include <QApplication>
#include <QCommandLineParser>

// Local includes
#include "settings.hpp"

class CommandLineOptions {
public:
  // Command-line options
  int nthreads;
  bool automaticThreads;
  float strength;
  float softening;
  int deviceIndex;
  int fps;
  bool debugGrid;
  int padFactor;
  int modelSize;
  float temporalSmooth;
  int personSensitivity;
  float lowerRes;
  std::string qualityMode;
  int secondsPerBackground;
  bool distortInside;
  bool flip;
  bool selectROI;
  std::string maskMode;
  std::string modelPath;
  std::string colorModeType;

  // Constructor is also the parser
  static CommandLineOptions parse(QApplication &app,
                                  const AppSettings &defaults) {
    QCommandLineParser parser;
    parser.setApplicationDescription(
        "GravyLensing applies a gravitational lensing effect to images based "
        "on people detected in a camera feed.");
    parser.addHelpOption();

    // --nthreads <int>
    QCommandLineOption nthreadsOption(
        QStringList() << "n" << "nthreads",
        "Number of CPU threads used in the calculation (must be >= 2).",
        "nthreads", QString::number(defaults.nthreads));
    parser.addOption(nthreadsOption);

    // --strength <float> (default 4.0)
    QCommandLineOption strengthOption(
        QStringList() << "s" << "strength",
        "Strength factor for the lensing effect (float, default=4.0).",
        "strength", QString::number(defaults.strength));
    parser.addOption(strengthOption);

    // --softening <float> (default 50.0)
    QCommandLineOption softeningOption(
        QStringList() << "f" << "softening",
        "Softening radius in pixels applied to the lensing effect (float, "
        "default=50.0).",
        "softening", QString::number(defaults.softening));
    parser.addOption(softeningOption);

    // --modelSize <int> (default 512)
    QCommandLineOption modelSizeOption(
        QStringList() << "m" << "modelSize",
        "Segmentation model size, bigger means more accurate people but at the "
        "expense of frame rate (int, default=512).",
        "modelSize", QString::number(defaults.modelSize));
    parser.addOption(modelSizeOption);

    // --device-index <int> (default 0)
    QCommandLineOption deviceIndexOption(
        QStringList() << "d" << "deviceIndex",
        "Device index, i.e. which camera to use (int, default=0).",
        "deviceIndex", QString::number(defaults.deviceIndex));
    parser.addOption(deviceIndexOption);

    // --fps <int> (default 30)
    QCommandLineOption fpsOption(
        QStringList() << "fps" << "frameRate",
        "Target camera frame rate (int, default=30).",
        "fps", QString::number(defaults.fps));
    parser.addOption(fpsOption);

    // --debug-grid  (flag only; no argument)
    QCommandLineOption debugGridOption(
        QStringList() << "g" << "debugGrid",
        "Show a debugging grid with the camera feed, mask, and lensed image.");
    parser.addOption(debugGridOption);
    QCommandLineOption noDebugGridOption(
        QStringList() << "no-debugGrid",
        "Force the debugging grid off for this session.");
    parser.addOption(noDebugGridOption);

    // --pad-factor <int> (default 2)
    QCommandLineOption padFactorOption(
        QStringList() << "p" << "padFactor",
        "Padding factor for FFT (int, default=2).", "padFactor",
        QString::number(defaults.padFactor));
    parser.addOption(padFactorOption);

    // --model-path <string>
    QCommandLineOption modelPathOption(
        QStringList() << "mp" << "modelPath",
        "Path to the segmentation model (string).", "modelPath",
        QString::fromStdString(defaults.modelPath));
    parser.addOption(modelPathOption);

    // --temporal smooth <float> (default is loaded from settings)
    QCommandLineOption temporalSmoothOption(
        QStringList() << "t" << "temporalSmooth",
        "Temporal frame smoothing factor, i.e. how much of previous frames is "
        "used to smooth out temporal flucations in the person detection mask "
        "(float, default=0.25).",
        "temporalSmooth", QString::number(defaults.temporalSmooth));
    parser.addOption(temporalSmoothOption);

    QCommandLineOption personSensitivityOption(
        QStringList() << "personSensitivity",
        "Person detection sensitivity from 0 (strict) to 100 (sensitive).",
        "personSensitivity", QString::number(defaults.personSensitivity));
    parser.addOption(personSensitivityOption);

    // lowerRes <float> (default 0.5)
    QCommandLineOption lowerResOption(
        QStringList() << "lr" << "lowerRes",
        "Lower resolution factor for the lensing effect (float, default=0.5).",
        "lowerRes", QString::number(defaults.lowerRes));
    parser.addOption(lowerResOption);

    QCommandLineOption qualityModeOption(
        QStringList() << "quality" << "qualityMode",
        "Quality preset for person mode: fast, balanced, high, or custom.",
        "qualityMode", QString::fromStdString(defaults.qualityMode));
    parser.addOption(qualityModeOption);

    // secondsPerBackground <int> (default -1, i.e infinite)
    QCommandLineOption secondsPerBackgroundOption(
        QStringList() << "sb" << "secondsPerBackground",
        "Seconds per background image, if -1 then background images are "
        "selected with the arrow keys (int, default=-1).",
        "secondsPerBackground", QString::number(defaults.secondsPerBackground));
    parser.addOption(secondsPerBackgroundOption);

    // distortInside <bool> (flag only; no argument)
    QCommandLineOption distortInsideOption(
        QStringList() << "di" << "distortInside", "Distort inside the mask?");
    parser.addOption(distortInsideOption);
    QCommandLineOption noDistortInsideOption(
        QStringList() << "no-distortInside",
        "Force interior distortion off for this session.");
    parser.addOption(noDistortInsideOption);

    // flip <bool> (flag only; no argument)
    QCommandLineOption flipOption(QStringList() << "flip",
                                  "Flip the camera feed horizontally?");
    parser.addOption(flipOption);
    QCommandLineOption noFlipOption(QStringList() << "no-flip",
                                    "Force camera mirroring off for this session.");
    parser.addOption(noFlipOption);

    // selectROI <bool> (flag only; no argument)
    QCommandLineOption selectROIOption(
        QStringList() << "roi" << "selectROI",
        "Select a region of interest (ROI) in the camera feed to apply the "
        "lensing effect. If not set, the full frame is used.");
    parser.addOption(selectROIOption);
    QCommandLineOption noSelectROIOption(
        QStringList() << "no-selectROI",
        "Skip the startup ROI selector for this session.");
    parser.addOption(noSelectROIOption);

    parser.process(app);

    bool ok;
    CommandLineOptions opts{};
    const auto resolveBool = [&](bool defaultValue,
                                 const QCommandLineOption &enableOption,
                                 const QCommandLineOption &disableOption,
                                 const char *name) {
      const bool enable = parser.isSet(enableOption);
      const bool disable = parser.isSet(disableOption);
      if (enable && disable) {
        std::cerr << "Error: --" << name << " and --"
                  << disableOption.names().constFirst().toStdString()
                  << " cannot be used together.\n";
        std::exit(-1);
      }
      if (enable)
        return true;
      if (disable)
        return false;
      return defaultValue;
    };

    opts.nthreads = parser.value(nthreadsOption).toInt(&ok);
    if (!ok || opts.nthreads < 2) {
      std::cerr << "Error: --nthreads must be an integer >= 2.\n";
      std::exit(-1);
    }
    opts.automaticThreads =
        parser.isSet(nthreadsOption) ? false : defaults.automaticThreads;

    opts.strength = parser.value(strengthOption).toFloat(&ok);
    if (!ok) {
      std::cerr << "Error: --strength must be a float.\n";
      std::exit(-1);
    }

    opts.softening = parser.value(softeningOption).toFloat(&ok);
    if (!ok) {
      std::cerr << "Error: --softening must be a float.\n";
      std::exit(-1);
    }

    opts.modelSize = parser.value(modelSizeOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --modelSize must be an integer.\n";
      std::exit(-1);
    }

    opts.deviceIndex = parser.value(deviceIndexOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --deviceIndex must be an integer.\n";
      std::exit(-1);
    }

    opts.fps = parser.value(fpsOption).toInt(&ok);
    if (!ok || opts.fps < 1) {
      std::cerr << "Error: --fps must be a positive integer.\n";
      std::exit(-1);
    }

    opts.debugGrid = resolveBool(defaults.debugGrid, debugGridOption,
                                 noDebugGridOption, "debugGrid");

    opts.padFactor = parser.value(padFactorOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --padFactor must be an integer.\n";
      std::exit(-1);
    }

    opts.modelPath = parser.value(modelPathOption).toStdString();
    if (opts.modelPath.empty()) {
      std::cerr << "Error: --modelPath must be a non-empty string.\n";
      std::exit(-1);
    }

    opts.temporalSmooth = parser.value(temporalSmoothOption).toFloat(&ok);
    if (!ok) {
      std::cerr << "Error: --temporalSmooth must be a float.\n";
      std::exit(-1);
    }

    opts.personSensitivity = parser.value(personSensitivityOption).toInt(&ok);
    if (!ok || opts.personSensitivity < 0 || opts.personSensitivity > 100) {
      std::cerr << "Error: --personSensitivity must be between 0 and 100.\n";
      std::exit(-1);
    }

    opts.lowerRes = parser.value(lowerResOption).toFloat(&ok);
    if (!ok) {
      std::cerr << "Error: --lowerRes must be a float.\n";
      std::exit(-1);
    }

    opts.qualityMode = parser.value(qualityModeOption).trimmed().toLower().toStdString();
    if (opts.qualityMode != "fast" && opts.qualityMode != "balanced" &&
        opts.qualityMode != "high" && opts.qualityMode != "custom") {
      std::cerr << "Error: --qualityMode must be one of fast, balanced, high, or custom.\n";
      std::exit(-1);
    }

    opts.secondsPerBackground =
        parser.value(secondsPerBackgroundOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --secondsPerBackground must be an integer.\n";
      std::exit(-1);
    }

    opts.distortInside = resolveBool(defaults.distortInside,
                                     distortInsideOption,
                                     noDistortInsideOption,
                                     "distortInside");
    opts.flip = resolveBool(defaults.flip, flipOption, noFlipOption, "flip");
    opts.selectROI = resolveBool(defaults.selectROI, selectROIOption,
                                 noSelectROIOption, "selectROI");
    opts.maskMode = defaults.maskMode;
    opts.colorModeType = defaults.colorModeType;

    return opts;
  }
};
