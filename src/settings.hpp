/**
 * @file settings.hpp
 *
 * Persistent application settings for GravyLensing.
 *
 * AppSettings holds every user-facing configuration value with sensible
 * defaults.  Settings are loaded from / saved to QSettings using the
 * load() and save() methods.  The equals() helper is used by the session
 * flow to detect no-op restarts.
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

#include <string>

#include <QCoreApplication>
#include <QSettings>

struct AppSettings {

  // ── Performance ────────────────────────────────────────────────────
  int nthreads = 12;            // Total CPU threads (Qt reserves 3)

  // ── Lensing ────────────────────────────────────────────────────────
  float strength = 4.0f;        // Deflection multiplier
  float softening = 50.0f;      // Kernel softening radius (px)
  int padFactor = 2;            // FFT padding factor
  float lowerRes = 0.5f;        // Resolution scale for lensing (0.1–1.0)
  bool distortInside = true;    // Lens the interior of the mask too

  // ── Camera ─────────────────────────────────────────────────────────
  int deviceIndex = 0;          // OpenCV camera device index
  int fps = 30;                 // Target camera frame rate
  bool flip = false;            // Mirror feed horizontally
  bool selectROI = false;       // Open ROI selector on first start

  // ── Mask mode ──────────────────────────────────────────────────────
  std::string maskMode = "person";   // "person" or "color"
  std::string colorModeType = "fixed_key";  // "fixed_key" or "tracked_blob"

  // ── Color key tolerances ───────────────────────────────────────────
  int colorHueTol = 12;     // ± tolerance around target hue (0-180)
  int colorSatTol = 60;     // ± tolerance around target saturation (0-255)
  int colorValTol = 80;     // ± tolerance around target value (0-255)

  // ── Person detection ───────────────────────────────────────────────
  std::string modelPath = "models/lraspp_torchscript-traced_float32_512_512.pt";
  int modelSize = 512;          // Segmentation model input size (px)
  float temporalSmooth = 0.25f; // Frame blending factor (0–1)
  std::string qualityMode = "balanced"; // fast, balanced, high, custom

  // ── Runtime ────────────────────────────────────────────────────────
  bool debugGrid = false;       // Show 2x2 diagnostic view
#ifdef __APPLE__
  std::string backgroundsDir =
      (QCoreApplication::applicationDirPath() + "/../Resources/backgrounds")
          .toStdString();
#else
  std::string backgroundsDir = "backgrounds/";
#endif
  int secondsPerBackground = -1;// Auto-cycle interval; -1 = manual

  /// True when every field matches.
  bool equals(const AppSettings &other) const {
    return nthreads == other.nthreads && strength == other.strength &&
           softening == other.softening && deviceIndex == other.deviceIndex &&
           fps == other.fps && debugGrid == other.debugGrid && padFactor == other.padFactor &&
           modelSize == other.modelSize &&
           temporalSmooth == other.temporalSmooth &&
           qualityMode == other.qualityMode &&
           lowerRes == other.lowerRes &&
           secondsPerBackground == other.secondsPerBackground &&
           distortInside == other.distortInside && flip == other.flip &&
           selectROI == other.selectROI && maskMode == other.maskMode &&
           colorModeType == other.colorModeType &&
           colorHueTol == other.colorHueTol &&
           colorSatTol == other.colorSatTol &&
           colorValTol == other.colorValTol &&
           backgroundsDir == other.backgroundsDir &&
           modelPath == other.modelPath;
  }

  /// Load from persistent storage, keeping current values as fallbacks.
  void load(QSettings &s) {
    nthreads = s.value("nthreads", nthreads).toInt();
    strength = s.value("strength", strength).toFloat();
    softening = s.value("softening", softening).toFloat();
    deviceIndex = s.value("deviceIndex", deviceIndex).toInt();
    fps = s.value("fps", fps).toInt();
    debugGrid = s.value("debugGrid", debugGrid).toBool();
    padFactor = s.value("padFactor", padFactor).toInt();
    modelSize = s.value("modelSize", modelSize).toInt();
    temporalSmooth = s.value("temporalSmooth", temporalSmooth).toFloat();
    qualityMode =
        s.value("qualityMode", QString::fromStdString(qualityMode)).toString().toStdString();
    lowerRes = s.value("lowerRes", lowerRes).toFloat();
    secondsPerBackground =
        s.value("secondsPerBackground", secondsPerBackground).toInt();
    distortInside = s.value("distortInside", distortInside).toBool();
    flip = s.value("flip", flip).toBool();
    selectROI = s.value("selectROI", selectROI).toBool();
    maskMode =
        s.value("maskMode", QString::fromStdString(maskMode)).toString().toStdString();
    colorModeType = s.value("colorModeType", QString::fromStdString(colorModeType))
                        .toString()
                        .toStdString();
    colorHueTol = s.value("colorHueTol", colorHueTol).toInt();
    colorSatTol = s.value("colorSatTol", colorSatTol).toInt();
    colorValTol = s.value("colorValTol", colorValTol).toInt();
    backgroundsDir =
        s.value("backgroundsDir", QString::fromStdString(backgroundsDir))
            .toString()
            .toStdString();
    modelPath =
        s.value("modelPath", QString::fromStdString(modelPath)).toString().toStdString();
  }

  /// Write all fields to persistent storage.
  void save(QSettings &s) const {
    s.setValue("nthreads", nthreads);
    s.setValue("strength", strength);
    s.setValue("softening", softening);
    s.setValue("deviceIndex", deviceIndex);
    s.setValue("fps", fps);
    s.setValue("debugGrid", debugGrid);
    s.setValue("padFactor", padFactor);
    s.setValue("modelSize", modelSize);
    s.setValue("temporalSmooth", temporalSmooth);
    s.setValue("qualityMode", QString::fromStdString(qualityMode));
    s.setValue("lowerRes", lowerRes);
    s.setValue("secondsPerBackground", secondsPerBackground);
    s.setValue("distortInside", distortInside);
    s.setValue("flip", flip);
    s.setValue("selectROI", selectROI);
    s.setValue("maskMode", QString::fromStdString(maskMode));
    s.setValue("colorModeType", QString::fromStdString(colorModeType));
    s.setValue("colorHueTol", colorHueTol);
    s.setValue("colorSatTol", colorSatTol);
    s.setValue("colorValTol", colorValTol);
    s.setValue("backgroundsDir", QString::fromStdString(backgroundsDir));
    s.setValue("modelPath", QString::fromStdString(modelPath));
  }

  AppSettings withQualityModeApplied() const {
    AppSettings tuned = *this;
    if (qualityMode == "fast") {
      tuned.modelSize = 224;
      tuned.temporalSmooth = 0.16f;
      tuned.lowerRes = 0.35f;
    } else if (qualityMode == "high") {
      tuned.modelSize = 640;
      tuned.temporalSmooth = 0.35f;
      tuned.lowerRes = 0.75f;
    } else if (qualityMode == "balanced") {
      tuned.modelSize = 512;
      tuned.temporalSmooth = 0.25f;
      tuned.lowerRes = 0.50f;
    }
    return tuned;
  }

  float lensMassBlurSigma() const {
    if (qualityMode == "fast")
      return 1.0f;
    if (qualityMode == "high")
      return 2.0f;
    return 1.5f;
  }

  std::string visionQualityMode() const {
    if (qualityMode == "fast")
      return "fast";
    if (qualityMode == "high")
      return "accurate";
    return "balanced";
  }
};
