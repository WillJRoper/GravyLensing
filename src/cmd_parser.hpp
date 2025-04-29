/**
 * CommandLineOptions
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

#include <QApplication>
#include <QCommandLineParser>
#include <iostream>

class CommandLineOptions {
public:
  // Command-line options
  int nthreads;
  float strength;
  float softening;
  int maskScale;
  int deviceIndex;
  bool debugGrid;
  int padFactor;
  std::string modelPath;

  // Constructor is also the parser
  static CommandLineOptions parse(QApplication &app) {
    QCommandLineParser parser;
    parser.setApplicationDescription("Your App Description");
    parser.addHelpOption();

    // --nthreads <int> (required)
    QCommandLineOption nthreadsOption(QStringList() << "n" << "nthreads",
                                      "Number of threads (must be >= 2).",
                                      "nthreads");
    parser.addOption(nthreadsOption);

    // --strength <float> (default 0.1)
    QCommandLineOption strengthOption(QStringList() << "s" << "strength",
                                      "Strength factor (float).", "strength",
                                      "0.1");
    parser.addOption(strengthOption);

    // --softening <float> (default 30.0)
    QCommandLineOption softeningOption(QStringList() << "f" << "softening",
                                       "Softening radius (float).", "softening",
                                       "30.0");
    parser.addOption(softeningOption);

    // --mask-scale <int> (default 4)
    QCommandLineOption maskScaleOption(QStringList() << "m" << "maskScale",
                                       "Mask scale factor (int).", "maskScale",
                                       "4");
    parser.addOption(maskScaleOption);

    // --device-index <int> (default 0)
    QCommandLineOption deviceIndexOption(QStringList() << "d" << "deviceIndex",
                                         "Device index (int).", "deviceIndex",
                                         "0");
    parser.addOption(deviceIndexOption);

    // --debug-grid  (flag only; no argument)
    QCommandLineOption debugGridOption(
        QStringList() << "g" << "debugGrid",
        "Show a debugging grid with the camera feed, mask, and lensed image.");
    parser.addOption(debugGridOption);

    // --pad-factor <int> (default 2)
    QCommandLineOption padFactorOption(QStringList() << "p" << "padFactor",
                                       "Padding factor for FFT (int).",
                                       "padFactor", "2");
    parser.addOption(padFactorOption);

    // --model-path <string> (default "models/deeplabv3_mobilenet_v3_large.pt")
    QCommandLineOption modelPathOption(
        QStringList() << "mp" << "modelPath",
        "Path to the segmentation model (string).", "modelPath",
        "models/deeplabv3_mobilenet_v3_large.pt");
    parser.addOption(modelPathOption);

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

    opts.maskScale = parser.value(maskScaleOption).toInt(&ok);
    if (!ok) {
      std::cerr << "Error: --maskScale must be an integer.\n";
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

    return opts;
  }
};
