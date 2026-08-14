/**
 * @file settings_dialog.hpp
 *
 * Settings dialog for GravyLensing — allows the user to configure all
 * application parameters through a GUI and persist them via QSettings.
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

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>

#include "settings.hpp"

class SettingsDialog : public QDialog {
  Q_OBJECT

public:
  /// @param settings     Initial values to seed the dialog controls.
  /// @param windowTitle  Dialog title (e.g. "Session Setup").
  /// @param acceptLabel  Text for the accept button (e.g. "Start Session").
  /// @param currentHue   Current selected target hue (0 if none).
  /// @param currentSat   Current selected target saturation (0 if none).
  /// @param currentVal   Current selected target value (0 if none).
  /// @param hasTarget    Whether a colour target is currently selected.
  /// @param hasROI       Whether a region of interest is currently set.
  /// @param roiX, roiY   Top-left corner of the current ROI.
  /// @param roiW, roiH   Width and height of the current ROI.
  /// @param parent       Parent widget.
  explicit SettingsDialog(const AppSettings &settings,
                          const QString &windowTitle = "Preferences",
                          const QString &acceptLabel = "OK",
                          float currentHue = 0, float currentSat = 0,
                          float currentVal = 0, bool hasTarget = false,
                          bool hasROI = false,
                          int roiX = 0, int roiY = 0,
                          int roiW = 0, int roiH = 0,
                          QWidget *parent = nullptr);

  /// Return the values currently set in the dialog controls.
  AppSettings settings() const;

  /// True if the user clicked the colour swatch to request a re-pick.
  bool colorPickRequested() const { return colorPickRequested_; }
  bool colorFramePickRequested() const { return colorFramePickRequested_; }
  float pickedHue() const { return pickedHue_; }
  float pickedSat() const { return pickedSat_; }
  float pickedVal() const { return pickedVal_; }

  /// True if the user clicked the "Select Region..." button.
  bool roiSelectRequested() const { return roiSelectRequested_; }
  bool roiClearRequested() const { return roiClearRequested_; }

private Q_SLOTS:
  void openColorPicker();

private:
  void updateSwatchDisplay(bool hasTarget);
  void updateBackgroundStatus();
#ifdef __APPLE__
  void refreshCameras();
#endif

private:
  QSpinBox *nthreadsSpin_;
  QCheckBox *automaticThreadsCheck_;
  QDoubleSpinBox *strengthSpin_;
  QDoubleSpinBox *softeningSpin_;
  QSpinBox *visionSizeSpin_;
#ifdef __APPLE__
  QComboBox *cameraCombo_;
#else
  QSpinBox *deviceIndexSpin_;
#endif
  QComboBox *fpsCombo_;
  QComboBox *qualityModeCombo_;
  QCheckBox *debugGridCheck_;
  QSpinBox *padFactorSpin_;
  QDoubleSpinBox *temporalSmoothSpin_;
  QSlider *personSensitivitySlider_;
  QDoubleSpinBox *lowerResSpin_;
  QCheckBox *distortInsideCheck_;
  QCheckBox *flipCheck_;
  QCheckBox *selectROICheck_;
  QRadioButton *personDetectionRadio_;
  QRadioButton *colorDetectionRadio_;
  QRadioButton *fixedColorRadio_;
  QRadioButton *trackedColorRadio_;
  QSpinBox *colorHueTolSpin_;
  QSpinBox *colorSatTolSpin_;
  QSpinBox *colorValTolSpin_;

  QLineEdit *backgroundsDirEdit_;
  QRadioButton *includedBackgroundsRadio_;
  QRadioButton *customBackgroundsRadio_;
  QLabel *backgroundStatus_;
  QPushButton *browseBgBtn_;
  QCheckBox *autoCycleCheck_;
  QSpinBox *secondsPerBackgroundSpin_;

  bool colorPickRequested_ = false;
  bool colorFramePickRequested_ = false;
  float pickedHue_ = 0;
  float pickedSat_ = 0;
  float pickedVal_ = 0;

  QPushButton *swatchBtn_ = nullptr;
  QLabel *swatchLabel_ = nullptr;

  bool roiSelectRequested_ = false;
  bool roiClearRequested_ = false;
  bool hasROI_ = false;
  int roiX_ = 0, roiY_ = 0, roiW_ = 0, roiH_ = 0;
  QLabel *roiInfoLabel_ = nullptr;
};

// Restore the macro if we undef'd it
#ifdef GRAVY_HAD_SLOTS_MACRO
#pragma pop_macro("slots")
#undef GRAVY_HAD_SLOTS_MACRO
#endif
