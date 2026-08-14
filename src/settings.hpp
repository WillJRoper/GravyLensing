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

#include <algorithm>
#include <string>

#include <QCoreApplication>
#include <QSettings>
#include <QThread>

struct AppSettings {

  static int defaultWorkerThreads() {
    return std::max(1, QThread::idealThreadCount() - 2);
  }

  // ── Performance ────────────────────────────────────────────────────
  int nthreads = defaultWorkerThreads(); // Leaves two logical cores available

  // ── Lensing ────────────────────────────────────────────────────────
  float strength = 4.0f;        // Deflection multiplier
  float softening = 50.0f;      // Kernel softening radius (px)
  int padFactor = 2;            // FFT padding factor
  float lowerRes = 0.5f;        // Internal calculation scale
  bool distortInside = true;    // Lens the interior of the mask too

  // ── Camera ─────────────────────────────────────────────────────────
  int deviceIndex = 0;          // OpenCV camera device index
  int fps = 30;                 // Target camera frame rate
  int cameraWidth = 1280;
  int cameraHeight = 720;
  bool flip = false;            // Mirror feed horizontally
  bool selectROI = false;       // Open ROI selector on first start

  // ── Mask mode ──────────────────────────────────────────────────────
  std::string maskMode = "person";   // "person" or "color"
  std::string colorModeType = "fixed_key";  // "fixed_key" or "tracked_blob"

  // ── Color key tolerances ───────────────────────────────────────────
  int colorHueTol = 12;     // ± tolerance around target hue (0-180)
  int colorSatTol = 60;     // ± tolerance around target saturation (0-255)
  int colorValTol = 80;     // ± tolerance around target value (0-255)
  int colorMinObjectArea = 500;
  int colorPersistenceFrames = 6;
  float colorMaskSmooth = 0.5f;

  // ── Person detection ───────────────────────────────────────────────
  int visionSize = 512;         // Vision request size (px)
  float temporalSmooth = 0.25f; // Frame blending factor (0–1)
  int personSensitivity = 50;   // Detection sensitivity (0 strict, 100 sensitive)
  std::string qualityMode = "balanced"; // fast, balanced, high, custom

  // ── Runtime ────────────────────────────────────────────────────────
  bool debugGrid = false;       // Show 2x2 diagnostic view
  bool showLensContents = false; // Composite camera pixels inside the mask
  int backgroundWidth = 1920;
  int backgroundHeight = 1080;
  std::string backgroundFitMode = "crop"; // crop, fit, stretch
  float lensEdgeSoftness = 1.5f;
  bool rebuildBackgroundCache = false; // Transient; never persisted
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
           fps == other.fps && cameraWidth == other.cameraWidth &&
           cameraHeight == other.cameraHeight &&
           debugGrid == other.debugGrid &&
           showLensContents == other.showLensContents &&
           padFactor == other.padFactor &&
           lowerRes == other.lowerRes &&
           visionSize == other.visionSize &&
           temporalSmooth == other.temporalSmooth &&
           personSensitivity == other.personSensitivity &&
           qualityMode == other.qualityMode &&
           secondsPerBackground == other.secondsPerBackground &&
           distortInside == other.distortInside && flip == other.flip &&
           selectROI == other.selectROI && maskMode == other.maskMode &&
           colorModeType == other.colorModeType &&
           colorHueTol == other.colorHueTol &&
           colorSatTol == other.colorSatTol &&
           colorValTol == other.colorValTol &&
           colorMinObjectArea == other.colorMinObjectArea &&
           colorPersistenceFrames == other.colorPersistenceFrames &&
           colorMaskSmooth == other.colorMaskSmooth &&
           backgroundsDir == other.backgroundsDir &&
           backgroundWidth == other.backgroundWidth &&
           backgroundHeight == other.backgroundHeight &&
           backgroundFitMode == other.backgroundFitMode &&
           lensEdgeSoftness == other.lensEdgeSoftness &&
           rebuildBackgroundCache == other.rebuildBackgroundCache;
  }

  /// Load from persistent storage, keeping current values as fallbacks.
  void load(QSettings &s) {
    const bool usedAutomaticThreads =
        s.value("automaticThreads", false).toBool();
    nthreads = usedAutomaticThreads
                   ? defaultWorkerThreads()
                   : s.value("nthreads", nthreads).toInt();
    s.remove("automaticThreads");
    strength = s.value("strength", strength).toFloat();
    softening = s.value("softening", softening).toFloat();
    deviceIndex = s.value("deviceIndex", deviceIndex).toInt();
    fps = s.value("fps", fps).toInt();
    cameraWidth = s.value("cameraWidth", cameraWidth).toInt();
    cameraHeight = s.value("cameraHeight", cameraHeight).toInt();
    debugGrid = s.value("debugGrid", debugGrid).toBool();
    showLensContents =
        s.value("showLensContents", showLensContents).toBool();
    padFactor = s.value("padFactor", padFactor).toInt();
    lowerRes = s.value("lowerRes", lowerRes).toFloat();
    visionSize = s.value("visionSize", visionSize).toInt();
    temporalSmooth = s.value("temporalSmooth", temporalSmooth).toFloat();
    personSensitivity =
        s.value("personSensitivity", personSensitivity).toInt();
    qualityMode =
        s.value("qualityMode", QString::fromStdString(qualityMode)).toString().toStdString();
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
    colorMinObjectArea =
        s.value("colorMinObjectArea", colorMinObjectArea).toInt();
    colorPersistenceFrames =
        s.value("colorPersistenceFrames", colorPersistenceFrames).toInt();
    colorMaskSmooth = s.value("colorMaskSmooth", colorMaskSmooth).toFloat();
    backgroundsDir =
        s.value("backgroundsDir", QString::fromStdString(backgroundsDir))
            .toString()
            .toStdString();
    backgroundWidth = s.value("backgroundWidth", backgroundWidth).toInt();
    backgroundHeight = s.value("backgroundHeight", backgroundHeight).toInt();
    backgroundFitMode =
        s.value("backgroundFitMode", QString::fromStdString(backgroundFitMode))
            .toString()
            .toStdString();
    lensEdgeSoftness = s.value("lensEdgeSoftness", lensEdgeSoftness).toFloat();
  }

  /// Write all fields to persistent storage.
  void save(QSettings &s) const {
    s.setValue("nthreads", nthreads);
    s.setValue("strength", strength);
    s.setValue("softening", softening);
    s.setValue("deviceIndex", deviceIndex);
    s.setValue("fps", fps);
    s.setValue("cameraWidth", cameraWidth);
    s.setValue("cameraHeight", cameraHeight);
    s.setValue("debugGrid", debugGrid);
    s.setValue("showLensContents", showLensContents);
    s.setValue("padFactor", padFactor);
    s.setValue("lowerRes", lowerRes);
    s.setValue("visionSize", visionSize);
    s.setValue("temporalSmooth", temporalSmooth);
    s.setValue("personSensitivity", personSensitivity);
    s.setValue("qualityMode", QString::fromStdString(qualityMode));
    s.setValue("secondsPerBackground", secondsPerBackground);
    s.setValue("distortInside", distortInside);
    s.setValue("flip", flip);
    s.setValue("selectROI", selectROI);
    s.setValue("maskMode", QString::fromStdString(maskMode));
    s.setValue("colorModeType", QString::fromStdString(colorModeType));
    s.setValue("colorHueTol", colorHueTol);
    s.setValue("colorSatTol", colorSatTol);
    s.setValue("colorValTol", colorValTol);
    s.setValue("colorMinObjectArea", colorMinObjectArea);
    s.setValue("colorPersistenceFrames", colorPersistenceFrames);
    s.setValue("colorMaskSmooth", colorMaskSmooth);
    s.setValue("backgroundsDir", QString::fromStdString(backgroundsDir));
    s.setValue("backgroundWidth", backgroundWidth);
    s.setValue("backgroundHeight", backgroundHeight);
    s.setValue("backgroundFitMode", QString::fromStdString(backgroundFitMode));
    s.setValue("lensEdgeSoftness", lensEdgeSoftness);
  }

  AppSettings withQualityModeApplied() const {
    AppSettings tuned = *this;
    if (qualityMode == "fast") {
      tuned.visionSize = 224;
      tuned.temporalSmooth = 0.16f;
      tuned.lowerRes = 0.35f;
    } else if (qualityMode == "high") {
      tuned.visionSize = 640;
      tuned.temporalSmooth = 0.35f;
      tuned.lowerRes = 0.75f;
    } else if (qualityMode == "balanced") {
      tuned.visionSize = 512;
      tuned.temporalSmooth = 0.25f;
      tuned.lowerRes = 0.50f;
    }
    return tuned;
  }

  float lensMassBlurSigma() const {
    return lensEdgeSoftness;
  }

  std::string visionQualityMode() const {
    if (qualityMode == "fast")
      return "fast";
    if (qualityMode == "high")
      return "accurate";
    return "balanced";
  }
};
