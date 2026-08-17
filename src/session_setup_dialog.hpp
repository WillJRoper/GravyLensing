#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QLineEdit>
#include <QLabel>
#include <QRadioButton>
#include <QSpinBox>

#include "settings.hpp"

class SessionSetupDialog : public QDialog {
  Q_OBJECT

public:
  SessionSetupDialog(const AppSettings &settings, float currentHue,
                     float currentSat, float currentVal, bool hasTarget,
                     bool hasROI, QWidget *parent = nullptr);

  AppSettings settings() const { return settings_; }
  bool colorPickRequested() const { return colorPickRequested_; }
  bool colorFramePickRequested() const { return colorFramePickRequested_; }
  float pickedHue() const { return pickedHue_; }
  float pickedSat() const { return pickedSat_; }
  float pickedVal() const { return pickedVal_; }

private:
  void applySettingsToControls();
  void updateSettingsFromControls();
  void updateBackgroundStatus();
#ifdef __APPLE__
  void refreshCameras();
#endif

  AppSettings settings_;
  float pickedHue_ = 0.0f;
  float pickedSat_ = 0.0f;
  float pickedVal_ = 0.0f;
  bool hasTarget_ = false;
  bool hasROI_ = false;
  bool colorPickRequested_ = false;
  bool colorFramePickRequested_ = false;

  QRadioButton *personRadio_ = nullptr;
  QRadioButton *colorRadio_ = nullptr;
#ifdef __APPLE__
  QComboBox *cameraCombo_ = nullptr;
#else
  QSpinBox *cameraSpin_ = nullptr;
#endif
  QComboBox *fpsCombo_ = nullptr;
  QComboBox *cameraResolutionCombo_ = nullptr;
  QSpinBox *customFpsSpin_ = nullptr;
  QComboBox *qualityCombo_ = nullptr;
  QCheckBox *mirrorCheck_ = nullptr;
  QCheckBox *selectRegionCheck_ = nullptr;
  QLineEdit *backgroundsEdit_ = nullptr;
  QRadioButton *includedBackgroundsRadio_ = nullptr;
  QRadioButton *customBackgroundsRadio_ = nullptr;
  QLabel *backgroundStatus_ = nullptr;
};
