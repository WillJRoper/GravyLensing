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

class CommandLineOptions {
public:
  // Command-line options
  int nthreads;
  float strength;
  float softening;
  int deviceIndex;
  bool debugGrid;
  int padFactor;
  int modelSize;
  float temporalSmooth;
  float lowerRes;
  int secondsPerBackground;
  bool distortInside;
  bool flip;
  bool selectROI;
  std::string modelPath;

  // Constructor is also the parser
  static CommandLineOptions parse(QApplication &app) {
    QCommandLineParser parser;
    parser.setApplicationDescription(
        "GravyLensing applies a gravitational lensing effect to images based "
        "on people detected in a camera feed.");
    parser.addHelpOption();

    // --nthreads <int> (required)
    QCommandLineOption nthreadsOption(
        QStringList() << "n" << "nthreads",
        "Number of CPU threads used in the calculation (must be >= 2).",
        "nthreads");
    parser.addOption(nthreadsOption);

    // --strength <float> (default 0.1)
    QCommandLineOption strengthOption(
        QStringList() << "s" << "strength",
        "Strength factor for the lensing effect (float, default=0.1).",
        "strength", "0.1");
    parser.addOption(strengthOption);

    // --softening <float> (default 30.0)
    QCommandLineOption softeningOption(
        QStringList() << "f" << "softening",
        "Softening radius in pixels applied to the lensing effect (float, "
        "default=30.0).",
        "softening", "30.0");
    parser.addOption(softeningOption);

    // --modelSize <int> (default 512)
    QCommandLineOption modelSizeOption(
        QStringList() << "m" << "modelSize",
        "Segmentation model size, bigger means more accurate people but at the "
        "expense of frame rate (int, default=512).",
        "modelSize", "512");
    parser.addOption(modelSizeOption);

    // --device-index <int> (default 0)
    QCommandLineOption deviceIndexOption(
        QStringList() << "d" << "deviceIndex",
        "Device index, i.e. which camera to use (int, default=0).",
        "deviceIndex", "0");
    parser.addOption(deviceIndexOption);

    // --debug-grid  (flag only; no argument)
    QCommandLineOption debugGridOption(
        QStringList() << "g" << "debugGrid",
        "Show a debugging grid with the camera feed, mask, and lensed image.");
    parser.addOption(debugGridOption);

    // --pad-factor <int> (default 2)
    QCommandLineOption padFactorOption(
        QStringList() << "p" << "padFactor",
        "Padding factor for FFT (int, default=2).", "padFactor", "2");
    parser.addOption(padFactorOption);

    // --model-path <string> (default "models/deeplabv3_mobilenet_v3_large.pt")
    QCommandLineOption modelPathOption(
        QStringList() << "mp" << "modelPath",
        "Path to the segmentation model (string).", "modelPath",
        "models/deeplab_quantized_model.pt");
    parser.addOption(modelPathOption);

    // --temporal smooth <float> (default 0.6)
    QCommandLineOption temporalSmoothOption(
        QStringList() << "t" << "temporalSmooth",
        "Temporal frame smoothing factor, i.e. how much of previous frames is "
        "used to smooth out temporal flucations in the person detection mask "
        "(float, default=0.25).",
        "temporalSmooth", "0.25");
    parser.addOption(temporalSmoothOption);

    // lowerRes <float> (default 1.0)
    QCommandLineOption lowerResOption(
        QStringList() << "lr" << "lowerRes",
        "Lower resolution factor for the lensing effect (float, default=1.0).",
        "lowerRes", "1.0");
    parser.addOption(lowerResOption);

    // secondsPerBackground <int> (default -1, i.e infinite)
    QCommandLineOption secondsPerBackgroundOption(
        QStringList() << "sb" << "secondsPerBackground",
        "Seconds per background image, if -1 then background images are "
        "selected through the 0-9 keys (int, default=-1).",
        "secondsPerBackground", "-1");
    parser.addOption(secondsPerBackgroundOption);

    // distortInside <bool> (flag only; no argument)
    QCommandLineOption distortInsideOption(
        QStringList() << "di" << "distortInside", "Distort inside the mask?");
    parser.addOption(distortInsideOption);

    // flip <bool> (flag only; no argument)
    QCommandLineOption flipOption(QStringList() << "flip",
                                  "Flip the camera feed horizontally?");
    parser.addOption(flipOption);

    // selectROI <bool> (flag only; no argument)
    QCommandLineOption selectROIOption(
        QStringList() << "roi" << "selectROI",
        "Select a region of interest (ROI) in the camera feed to apply the "
        "lensing effect. If not set, the full frame is used.");
    parser.addOption(selectROIOption);

    parser.process(app);

    // Validate required --nthreads
    if (!parser.isSet(nthreadsOption)) {
      std::cerr << "Error: --nthreads is required.\n";
      parser.showHelp(1);
    }

    bool ok;
    CommandLineOptions opts;
    opts.nthreads = parser.value(nthreadsOption).toInt(&ok);
    if (!ok || opts.nthreads < 2) {
      std::cerr << "Error: --nthreads must be an integer >= 2.\n";
      std::exit(-1);
    }

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

    opts.debugGrid = parser.isSet(debugGridOption);

    opts.padFactor = parser.value(padFactorOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --padFactor must be a float.\n";
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

    opts.lowerRes = parser.value(lowerResOption).toFloat(&ok);
    if (!ok) {
      std::cerr << "Error: --lowerRes must be a float.\n";
      std::exit(-1);
    }

    opts.secondsPerBackground =
        parser.value(secondsPerBackgroundOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --secondsPerBackground must be an integer.\n";
      std::exit(-1);
    }

    opts.distortInside = parser.isSet(distortInsideOption);
    opts.flip = parser.isSet(flipOption);
    opts.selectROI = parser.isSet(selectROIOption);

    return opts;
  }
};
