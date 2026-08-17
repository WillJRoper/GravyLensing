/**
 * @file settings_dialog.cpp
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

#include "settings_dialog.hpp"

#include <QColorDialog>
#include <QButtonGroup>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFont>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QSignalBlocker>
#include <QScrollArea>
#include <QSlider>
#include <QSizePolicy>
#include <QStackedWidget>
#include <QTimer>
#include <QVBoxLayout>

#ifdef __APPLE__
#include "avfoundation_camera.hpp"
#endif

#include "backgrounds.hpp"

static QDoubleSpinBox *makeDoubleSpin(double min, double max, double step,
                                       int decimals, const QString &suffix,
                                       const QString &tooltip) {
  auto *s = new QDoubleSpinBox;
  s->setRange(min, max);
  s->setSingleStep(step);
  s->setDecimals(decimals);
  s->setAccelerated(true);
  s->setKeyboardTracking(false);
  s->setMinimumWidth(120);
  if (!suffix.isEmpty())
    s->setSuffix(suffix);
  if (!tooltip.isEmpty())
    s->setToolTip(tooltip);
  return s;
}

static QSpinBox *makeIntSpin(int min, int max, int step, const QString &suffix,
                             const QString &tooltip) {
  auto *s = new QSpinBox;
  s->setRange(min, max);
  s->setSingleStep(step);
  s->setAccelerated(true);
  s->setKeyboardTracking(false);
  s->setMinimumWidth(120);
  if (!suffix.isEmpty())
    s->setSuffix(suffix);
  if (!tooltip.isEmpty())
    s->setToolTip(tooltip);
  return s;
}

static QCheckBox *makeCheck(const QString &text, const QString &tooltip) {
  auto *c = new QCheckBox(text);
  if (!tooltip.isEmpty())
    c->setToolTip(tooltip);
  return c;
}

static QLabel *makeFormLabel(const QString &text, const QString &tooltip,
                              QWidget *buddy = nullptr) {
  auto *label = new QLabel(text);
  label->setToolTip(tooltip);
  if (buddy != nullptr)
    label->setBuddy(buddy);
  return label;
}

static QLabel *makeDescription(const QString &text) {
  auto *label = new QLabel(text);
  label->setWordWrap(true);
  label->setForegroundRole(QPalette::PlaceholderText);
  label->setContentsMargins(24, 0, 0, 4);
  return label;
}

static void configureFormLayout(QFormLayout *form) {
  form->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);
  form->setFormAlignment(Qt::AlignTop);
  form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
  form->setRowWrapPolicy(QFormLayout::WrapLongRows);
  form->setHorizontalSpacing(14);
  form->setVerticalSpacing(10);
}

static void addFormRow(QFormLayout *form, const QString &label,
                       const QString &tooltip, QWidget *field) {
  form->addRow(makeFormLabel(label, tooltip, field), field);
}

static void addFormRow(QFormLayout *form, const QString &label,
                       const QString &tooltip, QLayout *fieldLayout) {
  form->addRow(makeFormLabel(label, tooltip), fieldLayout);
}

static QVBoxLayout *addSettingsPage(QStackedWidget *pages,
                                    const QString &title,
                                    const QString &summary) {
  auto *scrollArea = new QScrollArea;
  scrollArea->setWidgetResizable(true);
  scrollArea->setFrameShape(QFrame::NoFrame);

  auto *content = new QWidget;
  auto *layout = new QVBoxLayout(content);
  layout->setContentsMargins(24, 18, 24, 24);
  layout->setSpacing(16);

  auto *heading = new QLabel(title);
  QFont headingFont = heading->font();
  headingFont.setPointSize(22);
  headingFont.setWeight(QFont::DemiBold);
  heading->setFont(headingFont);
  layout->addWidget(heading);

  auto *description = new QLabel(summary);
  description->setWordWrap(true);
  description->setForegroundRole(QPalette::PlaceholderText);
  layout->addWidget(description);

  auto *separator = new QFrame;
  separator->setFrameShape(QFrame::HLine);
  layout->addWidget(separator);

  scrollArea->setWidget(content);
  pages->addWidget(scrollArea);
  return layout;
}

SettingsDialog::SettingsDialog(const AppSettings &settings,
                               const QString &windowTitle,
                               const QString &acceptLabel,
                               float currentHue, float currentSat,
                               float currentVal, bool hasTarget,
                               bool hasROI,
                               int roiX, int roiY, int roiW, int roiH,
                               QWidget *parent)
    : QDialog(parent) {

  setWindowTitle(windowTitle);
  setMinimumWidth(820);
  resize(960, 720);
  setSizeGripEnabled(true);

  auto *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(14, 14, 14, 14);
  mainLayout->setSpacing(12);

  auto *body = new QHBoxLayout;
  body->setSpacing(12);
  mainLayout->addLayout(body, 1);

  auto *sidebar = new QListWidget;
  sidebar->setObjectName("settingsSidebar");
  sidebar->setFixedWidth(190);
  sidebar->setSpacing(2);
  sidebar->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  sidebar->addItems({"Lens", "Camera & Region", "Lensing Effect",
                     "Backgrounds", "Advanced"});
  sidebar->setStyleSheet(
      "QListWidget#settingsSidebar { background: palette(alternate-base); "
      "border: 0; border-radius: 12px; padding: 8px; outline: 0; }"
      "QListWidget#settingsSidebar::item { border-radius: 7px; padding: 10px "
      "12px; }"
      "QListWidget#settingsSidebar::item:selected { "
      "background: palette(highlight); color: palette(highlighted-text); }");
  body->addWidget(sidebar);

  auto *pages = new QStackedWidget;
  body->addWidget(pages, 1);

  auto *subjectPage = addSettingsPage(
      pages, "Lens",
      "Choose what acts as the lens and tune how it is detected.");
  auto *cameraPage = addSettingsPage(
      pages, "Camera & Region",
      "Select the camera, preview behavior, frame rate, and active image area.");
  auto *effectPage = addSettingsPage(
      pages, "Lensing Effect",
      "Shape the strength, width, and appearance of the gravitational lens.");
  auto *backgroundsPage = addSettingsPage(
      pages, "Backgrounds",
      "Choose the images seen through the lens and control automatic cycling.");
  auto *advancedPage = addSettingsPage(
      pages, "Advanced",
      "Performance and diagnostic controls. Recommended defaults suit most Macs.");

  connect(sidebar, &QListWidget::currentRowChanged, pages,
          &QStackedWidget::setCurrentIndex);
  sidebar->setCurrentRow(0);

  // ── Camera group ────────────────────────────────────────────────────
  auto *cameraGroup = new QGroupBox("Camera");
  auto *cameraForm = new QFormLayout(cameraGroup);
  configureFormLayout(cameraForm);

#ifdef __APPLE__
  cameraCombo_ = new QComboBox;
  cameraCombo_->setToolTip(
      "Camera used for the live feed. Built-in and connected cameras are "
      "listed by their macOS names.");
  refreshCameras();
  const int savedCamera = cameraCombo_->findData(settings.deviceIndex);
  if (savedCamera >= 0)
    cameraCombo_->setCurrentIndex(savedCamera);
  auto *cameraTimer = new QTimer(this);
  connect(cameraTimer, &QTimer::timeout, this,
          &SettingsDialog::refreshCameras);
  cameraTimer->start(1000);
  addFormRow(cameraForm, "Camera", cameraCombo_->toolTip(), cameraCombo_);
#else
  deviceIndexSpin_ = makeIntSpin(0, 99, 1, {},
      "Camera 0 is normally the built-in camera. Try 1 or 2 only when the "
      "wrong camera opens.");
  deviceIndexSpin_->setValue(settings.deviceIndex);
  addFormRow(cameraForm, "Camera number", deviceIndexSpin_->toolTip(),
             deviceIndexSpin_);
#endif

  cameraResolutionCombo_ = new QComboBox;
  cameraResolutionCombo_->setToolTip(
      "Requested camera capture resolution. The closest format supported by "
      "the selected camera is used; actual format appears in the session "
      "window title.");
  cameraResolutionCombo_->addItem("Automatic", QSize());
  cameraResolutionCombo_->addItem("480p - 854 x 480", QSize(854, 480));
  cameraResolutionCombo_->addItem("720p - 1280 x 720 (Recommended)",
                                  QSize(1280, 720));
  cameraResolutionCombo_->addItem("1080p - 1920 x 1080", QSize(1920, 1080));
  const int cameraResolution = cameraResolutionCombo_->findData(
      QSize(settings.cameraWidth, settings.cameraHeight));
  cameraResolutionCombo_->setCurrentIndex(cameraResolution >= 0
                                               ? cameraResolution
                                               : 0);
  addFormRow(cameraForm, "Capture resolution",
             cameraResolutionCombo_->toolTip(), cameraResolutionCombo_);

  fpsCombo_ = new QComboBox;
  fpsCombo_->setToolTip(
      "Requested camera frame rate. 30 fps is recommended; lower it if the "
      "effect stutters, or use 60 fps only with a capable camera.");
  const int fpsOptions[] = {5, 10, 15, 20, 24, 25, 30, 60};
  for (const int f : fpsOptions) {
    fpsCombo_->addItem(QString("%1 fps").arg(f), f);
    if (f == settings.fps)
      fpsCombo_->setCurrentIndex(fpsCombo_->count() - 1);
  }
  addFormRow(cameraForm, "Frame rate", fpsCombo_->toolTip(), fpsCombo_);

  flipCheck_ = makeCheck("Mirror camera feed horizontally",
      "Flip the image so movement in the real world and on-screen are "
      "directionally consistent.");
  flipCheck_->setChecked(settings.flip);
  addFormRow(cameraForm, "Flip", flipCheck_->toolTip(), flipCheck_);

  selectROICheck_ = makeCheck("Show region selector at startup",
      "Ask you to select a camera region when the session starts. Leave this "
      "off to use the full frame.");
  selectROICheck_->setChecked(settings.selectROI);

  cameraPage->addWidget(cameraGroup);

  // ── Region of Interest ────────────────────────────────────────────
  auto *roiGroup = new QGroupBox("Region of Interest");
  auto *roiLayout = new QVBoxLayout(roiGroup);
  hasROI_ = hasROI;
  roiX_ = roiX; roiY_ = roiY; roiW_ = roiW; roiH_ = roiH;

  roiInfoLabel_ = new QLabel;
  roiInfoLabel_->setWordWrap(true);
  if (hasROI_) {
    roiInfoLabel_->setText(
        QString("Current region: (%1, %2)  %3 \u00D7 %4 px")
            .arg(roiX_).arg(roiY_).arg(roiW_).arg(roiH_));
  } else {
    roiInfoLabel_->setText("No region selected — full frame in use.");
  }
  roiLayout->addWidget(roiInfoLabel_);

  auto *roiBtnRow = new QHBoxLayout;
  auto *selectRoiBtn = new QPushButton("Select New Region from Camera...");
  selectRoiBtn->setToolTip(
      "Open the interactive region selector on the camera feed.");
  connect(selectRoiBtn, &QPushButton::clicked, this, [this]() {
    roiSelectRequested_ = true;
    accept();
  });
  roiBtnRow->addWidget(selectRoiBtn);

  auto *clearRoiBtn = new QPushButton("Use Full Frame");
  clearRoiBtn->setToolTip(
      "Remove the current region and use the entire camera image.");
  clearRoiBtn->setEnabled(hasROI_);
  connect(clearRoiBtn, &QPushButton::clicked, this, [this]() {
    hasROI_ = false;
    roiClearRequested_ = true;
    roiX_ = roiY_ = roiW_ = roiH_ = 0;
    roiInfoLabel_->setText("No region selected — full frame in use.");
    auto *btn = qobject_cast<QPushButton *>(sender());
    if (btn) btn->setEnabled(false);
  });
  roiBtnRow->addWidget(clearRoiBtn);
  roiLayout->addLayout(roiBtnRow);
  cameraPage->addWidget(roiGroup);
  cameraPage->addStretch(1);

  // ── Mode (mask type selection) ─────────────────────────────────────
  auto *modeGroup = new QGroupBox("Lens Detection");
  auto *modeLayout = new QVBoxLayout(modeGroup);
  auto *modeButtons = new QButtonGroup(modeGroup);

  personDetectionRadio_ = new QRadioButton("People (Recommended)");
  personDetectionRadio_->setToolTip(
      "Automatically detect people with Apple Vision. No colour selection is "
      "required.");
  modeButtons->addButton(personDetectionRadio_);
  modeLayout->addWidget(personDetectionRadio_);

  auto *personDescription = new QLabel(
      "Automatically finds people in the camera image. Best starting point "
      "and requires no manual setup.");
  personDescription->setWordWrap(true);
  personDescription->setForegroundRole(QPalette::PlaceholderText);
  personDescription->setContentsMargins(24, 0, 0, 8);
  personDescription->setToolTip(personDetectionRadio_->toolTip());
  modeLayout->addWidget(personDescription);

  colorDetectionRadio_ = new QRadioButton("A selected colour");
  colorDetectionRadio_->setToolTip(
      "Lens pixels matching a colour selected from the camera. Useful for "
      "objects, clothing, props, and green-screen-style effects.");
  modeButtons->addButton(colorDetectionRadio_);
  modeLayout->addWidget(colorDetectionRadio_);

  auto *colorDescription = new QLabel(
      "Choose a colour from the live camera after starting. Best for tracking "
      "an object or coloured area instead of a person.");
  colorDescription->setWordWrap(true);
  colorDescription->setForegroundRole(QPalette::PlaceholderText);
  colorDescription->setContentsMargins(24, 0, 0, 0);
  colorDescription->setToolTip(colorDetectionRadio_->toolTip());
  modeLayout->addWidget(colorDescription);

  colorDetectionRadio_->setChecked(settings.maskMode == "color");
  personDetectionRadio_->setChecked(settings.maskMode != "color");
  subjectPage->addWidget(modeGroup);

  // ── Person Detection group ──────────────────────────────────────────
  auto *personGroup = new QGroupBox("Person Detection");
  auto *personForm = new QFormLayout(personGroup);
  configureFormLayout(personForm);

  visionSizeSpin_ = makeIntSpin(128, 1024, 128, " px",
      "Advanced: working resolution used for person detection. Larger values "
      "can improve edges but reduce frame rate.");
  visionSizeSpin_->setValue(settings.visionSize);
  auto *visionSizeSlider = new QSlider(Qt::Horizontal);
  visionSizeSlider->setRange(128, 1024);
  visionSizeSlider->setSingleStep(32);
  visionSizeSlider->setPageStep(128);
  visionSizeSlider->setValue(settings.visionSize);
  visionSizeSlider->setToolTip(visionSizeSpin_->toolTip());
  auto *visionSizeLabel = new QLabel(QString("%1 px").arg(settings.visionSize));
  visionSizeLabel->setMinimumWidth(54);
  auto *visionSizeRow = new QHBoxLayout;
  visionSizeRow->addWidget(visionSizeSlider, 1);
  visionSizeRow->addWidget(visionSizeLabel);
  addFormRow(personForm, "Detection detail", visionSizeSpin_->toolTip(),
             visionSizeRow);
  connect(visionSizeSlider, &QSlider::valueChanged, visionSizeSpin_,
          &QSpinBox::setValue);
  connect(visionSizeSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [visionSizeSlider, visionSizeLabel](int value) {
            visionSizeSlider->setValue(value);
            visionSizeLabel->setText(QString("%1 px").arg(value));
          });

  qualityModeCombo_ = new QComboBox;
  qualityModeCombo_->setToolTip(
      "Controls the quality and speed of person detection. Balanced is the "
      "recommended starting point for Apple Silicon Macs.");
  qualityModeCombo_->addItem("Fast", "fast");
  qualityModeCombo_->addItem("Balanced (Recommended)", "balanced");
  qualityModeCombo_->addItem("High Quality", "high");
  qualityModeCombo_->addItem("Custom", "custom");
  {
    const int idx = qualityModeCombo_->findData(
        QString::fromStdString(settings.qualityMode));
    qualityModeCombo_->setCurrentIndex(idx >= 0 ? idx : 1);
  }
  addFormRow(personForm, "Quality mode", qualityModeCombo_->toolTip(),
             qualityModeCombo_);

  temporalSmoothSpin_ = makeDoubleSpin(0.0, 1.0, 0.05, 2, {},
      "Advanced: higher values reduce mask flicker but make detection respond "
      "more slowly to movement.");
  temporalSmoothSpin_->setValue(static_cast<double>(settings.temporalSmooth));
  auto *stabilizationSlider = new QSlider(Qt::Horizontal);
  stabilizationSlider->setRange(0, 100);
  stabilizationSlider->setSingleStep(5);
  stabilizationSlider->setValue(
      static_cast<int>(std::round(settings.temporalSmooth * 100.0f)));
  stabilizationSlider->setToolTip(temporalSmoothSpin_->toolTip());
  auto *stabilizationLabel =
      new QLabel(QString("%1%").arg(stabilizationSlider->value()));
  stabilizationLabel->setMinimumWidth(40);
  auto *stabilizationRow = new QHBoxLayout;
  stabilizationRow->addWidget(stabilizationSlider, 1);
  stabilizationRow->addWidget(stabilizationLabel);
  addFormRow(personForm, "Mask stability", temporalSmoothSpin_->toolTip(),
             stabilizationRow);
  connect(stabilizationSlider, &QSlider::valueChanged, this,
          [this](int value) {
            temporalSmoothSpin_->setValue(static_cast<double>(value) / 100.0);
          });
  connect(temporalSmoothSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged),
          this, [stabilizationSlider, stabilizationLabel](double value) {
            const int percent = static_cast<int>(std::round(value * 100.0));
            stabilizationSlider->setValue(percent);
            stabilizationLabel->setText(QString("%1%").arg(percent));
          });

  personSensitivitySlider_ = new QSlider(Qt::Horizontal);
  personSensitivitySlider_->setRange(0, 100);
  personSensitivitySlider_->setValue(settings.personSensitivity);
  personSensitivitySlider_->setToolTip(
      "Increase to detect smaller, distant, or partially visible people. "
      "Decrease if background objects are detected as people.");
  auto *sensitivityLabel = new QLabel;
  sensitivityLabel->setMinimumWidth(62);
  const auto updateSensitivityLabel = [sensitivityLabel](int value) {
    sensitivityLabel->setText(value < 35 ? "Strict"
                              : value > 65 ? "Sensitive"
                                           : "Standard");
  };
  updateSensitivityLabel(settings.personSensitivity);
  auto *sensitivityRow = new QHBoxLayout;
  sensitivityRow->addWidget(personSensitivitySlider_, 1);
  sensitivityRow->addWidget(sensitivityLabel);
  addFormRow(personForm, "Sensitivity", personSensitivitySlider_->toolTip(),
             sensitivityRow);
  connect(personSensitivitySlider_, &QSlider::valueChanged, this,
          updateSensitivityLabel);

  subjectPage->addWidget(personGroup);

  const auto applyQualityPreset = [this](const QString &mode) {
    if (mode == QLatin1String("fast")) {
      visionSizeSpin_->setValue(224);
      temporalSmoothSpin_->setValue(0.16);
      lowerResSpin_->setValue(0.35);
    } else if (mode == QLatin1String("balanced")) {
      visionSizeSpin_->setValue(512);
      temporalSmoothSpin_->setValue(0.25);
      lowerResSpin_->setValue(0.50);
    } else if (mode == QLatin1String("high")) {
      visionSizeSpin_->setValue(640);
      temporalSmoothSpin_->setValue(0.35);
      lowerResSpin_->setValue(0.75);
    }
  };

  const auto syncQualityModeFromControls = [this]() {
    const auto roughlyEqual = [](double a, double b) {
      return std::abs(a - b) < 0.01;
    };
    QString mode = "custom";
    if (visionSizeSpin_->value() == 224 &&
        roughlyEqual(temporalSmoothSpin_->value(), 0.16) &&
        roughlyEqual(lowerResSpin_->value(), 0.35)) {
      mode = "fast";
    } else if (visionSizeSpin_->value() == 512 &&
               roughlyEqual(temporalSmoothSpin_->value(), 0.25) &&
               roughlyEqual(lowerResSpin_->value(), 0.50)) {
      mode = "balanced";
    } else if (visionSizeSpin_->value() == 640 &&
               roughlyEqual(temporalSmoothSpin_->value(), 0.35) &&
               roughlyEqual(lowerResSpin_->value(), 0.75)) {
      mode = "high";
    }
    const int idx = qualityModeCombo_->findData(mode);
    if (idx >= 0 && qualityModeCombo_->currentIndex() != idx) {
      QSignalBlocker blocker(qualityModeCombo_);
      qualityModeCombo_->setCurrentIndex(idx);
    }
  };

  connect(qualityModeCombo_, &QComboBox::currentIndexChanged, this,
          [this, applyQualityPreset](int) {
            const QString mode = qualityModeCombo_->currentData().toString();
            if (mode == QLatin1String("custom")) {
              return;
            }
            applyQualityPreset(mode);
          });
  connect(visionSizeSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [syncQualityModeFromControls](int) { syncQualityModeFromControls(); });
  connect(temporalSmoothSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged),
          this, [syncQualityModeFromControls](double) {
            syncQualityModeFromControls();
          });

  // ── Color Detection (subtype + tolerances) ─────────────────────────
  auto *colorGroup = new QGroupBox("Color Detection");
  auto *colorLayout = new QVBoxLayout(colorGroup);
  auto *colorForm = new QFormLayout;
  configureFormLayout(colorForm);

  auto *colorModeButtons = new QButtonGroup(colorGroup);
  fixedColorRadio_ = new QRadioButton("Match every pixel (Recommended)");
  fixedColorRadio_->setToolTip(
      "Lens every pixel matching the selected colour. Best for coloured areas, "
      "clothing, and green-screen-style effects.");
  trackedColorRadio_ = new QRadioButton("Track one coloured object");
  trackedColorRadio_->setToolTip(
      "Follow one connected object matching the selected colour. Use when other "
      "parts of the scene contain similar colours.");
  colorModeButtons->addButton(fixedColorRadio_);
  colorModeButtons->addButton(trackedColorRadio_);
  fixedColorRadio_->setChecked(settings.colorModeType != "tracked_blob");
  trackedColorRadio_->setChecked(settings.colorModeType == "tracked_blob");
  colorLayout->addWidget(fixedColorRadio_);
  colorLayout->addWidget(makeDescription(
      "Lenses all matching areas. Most predictable and easiest to set up."));
  colorLayout->addWidget(trackedColorRadio_);
  colorLayout->addWidget(makeDescription(
      "Follows one object when the scene contains several similar colours."));

  colorHueTolSpin_ = makeIntSpin(1, 90, 1, {},
      "How much colour variation to accept. Increase when parts of the object "
      "are missed; decrease when unrelated colours are included.");
  colorHueTolSpin_->setValue(settings.colorHueTol);
  auto *hueSlider = new QSlider(Qt::Horizontal);
  hueSlider->setRange(1, 90);
  hueSlider->setValue(settings.colorHueTol);
  hueSlider->setToolTip(colorHueTolSpin_->toolTip());
  auto *hueLabel = new QLabel;
  hueLabel->setMinimumWidth(56);
  const auto updateHueLabel = [hueLabel](int value) {
    hueLabel->setText(value < 10 ? "Narrow" : value > 24 ? "Wide" : "Standard");
  };
  updateHueLabel(settings.colorHueTol);
  auto *hueRow = new QHBoxLayout;
  hueRow->addWidget(hueSlider, 1);
  hueRow->addWidget(hueLabel);
  addFormRow(colorForm, "Colour range", hueSlider->toolTip(), hueRow);
  connect(hueSlider, &QSlider::valueChanged, colorHueTolSpin_,
          &QSpinBox::setValue);
  connect(hueSlider, &QSlider::valueChanged, this, updateHueLabel);
  connect(colorHueTolSpin_, qOverload<int>(&QSpinBox::valueChanged), hueSlider,
          &QSlider::setValue);

  colorSatTolSpin_ = makeIntSpin(1, 255, 5, {},
      "How much vividness variation to accept. Increase for uneven lighting; "
      "decrease when grey areas are included.");
  colorSatTolSpin_->setValue(settings.colorSatTol);
  auto *saturationSlider = new QSlider(Qt::Horizontal);
  saturationSlider->setRange(1, 255);
  saturationSlider->setValue(settings.colorSatTol);
  saturationSlider->setToolTip(colorSatTolSpin_->toolTip());
  auto *saturationLabel = new QLabel;
  saturationLabel->setMinimumWidth(56);
  const auto updateSaturationLabel = [saturationLabel](int value) {
    saturationLabel->setText(value < 45 ? "Narrow"
                             : value > 100 ? "Wide"
                                           : "Standard");
  };
  updateSaturationLabel(settings.colorSatTol);
  auto *saturationRow = new QHBoxLayout;
  saturationRow->addWidget(saturationSlider, 1);
  saturationRow->addWidget(saturationLabel);
  addFormRow(colorForm, "Vividness range", saturationSlider->toolTip(),
             saturationRow);
  connect(saturationSlider, &QSlider::valueChanged, colorSatTolSpin_,
          &QSpinBox::setValue);
  connect(saturationSlider, &QSlider::valueChanged, this,
          updateSaturationLabel);
  connect(colorSatTolSpin_, qOverload<int>(&QSpinBox::valueChanged),
          saturationSlider, &QSlider::setValue);

  colorValTolSpin_ = makeIntSpin(1, 255, 5, {},
      "How much brightness variation to accept. Increase when shadows or "
      "highlights create holes in the mask.");
  colorValTolSpin_->setValue(settings.colorValTol);
  auto *brightnessSlider = new QSlider(Qt::Horizontal);
  brightnessSlider->setRange(1, 255);
  brightnessSlider->setValue(settings.colorValTol);
  brightnessSlider->setToolTip(colorValTolSpin_->toolTip());
  auto *brightnessLabel = new QLabel;
  brightnessLabel->setMinimumWidth(56);
  const auto updateBrightnessLabel = [brightnessLabel](int value) {
    brightnessLabel->setText(value < 60 ? "Narrow"
                             : value > 120 ? "Wide"
                                           : "Standard");
  };
  updateBrightnessLabel(settings.colorValTol);
  auto *brightnessRow = new QHBoxLayout;
  brightnessRow->addWidget(brightnessSlider, 1);
  brightnessRow->addWidget(brightnessLabel);
  addFormRow(colorForm, "Brightness range", brightnessSlider->toolTip(),
             brightnessRow);
  connect(brightnessSlider, &QSlider::valueChanged, colorValTolSpin_,
          &QSpinBox::setValue);
  connect(brightnessSlider, &QSlider::valueChanged, this,
          updateBrightnessLabel);
  connect(colorValTolSpin_, qOverload<int>(&QSpinBox::valueChanged),
          brightnessSlider, &QSlider::setValue);

  colorMinObjectAreaSpin_ = makeIntSpin(
      20, 100000, 100, " px",
      "Ignore matching objects smaller than this processed-mask area. Increase "
      "to reject speckles; decrease to track small or distant objects.");
  colorMinObjectAreaSpin_->setValue(settings.colorMinObjectArea);
  auto *minObjectRow = new QHBoxLayout;
  minObjectRow->addWidget(colorMinObjectAreaSpin_);
  minObjectRow->addStretch(1);
  addFormRow(colorForm, "Minimum object size",
             colorMinObjectAreaSpin_->toolTip(), minObjectRow);

  colorPersistenceSpin_ = makeIntSpin(
      0, 60, 1, " frames",
      "Keep the last tracked mask briefly when the object disappears. Higher "
      "values bridge occlusion but can leave a stale mask.");
  colorPersistenceSpin_->setValue(settings.colorPersistenceFrames);
  auto *persistenceRow = new QHBoxLayout;
  persistenceRow->addWidget(colorPersistenceSpin_);
  persistenceRow->addStretch(1);
  addFormRow(colorForm, "Tracking persistence",
             colorPersistenceSpin_->toolTip(), persistenceRow);

  colorMaskStabilitySlider_ = new QSlider(Qt::Horizontal);
  colorMaskStabilitySlider_->setRange(0, 100);
  colorMaskStabilitySlider_->setValue(static_cast<int>(
      std::round((1.0f - settings.colorMaskSmooth) * 100.0f)));
  colorMaskStabilitySlider_->setToolTip(
      "Higher values smooth frame-to-frame mask changes. Lower values respond "
      "faster to movement.");
  auto *stabilityLabel = new QLabel;
  const auto updateColorStabilityLabel = [stabilityLabel](int value) {
    stabilityLabel->setText(value < 35 ? "Responsive"
                           : value > 65 ? "Stable"
                                        : "Balanced");
  };
  updateColorStabilityLabel(colorMaskStabilitySlider_->value());
  auto *colorStabilityRow = new QHBoxLayout;
  colorStabilityRow->addWidget(colorMaskStabilitySlider_, 1);
  colorStabilityRow->addWidget(stabilityLabel);
  addFormRow(colorForm, "Mask stability",
             colorMaskStabilitySlider_->toolTip(), colorStabilityRow);
  connect(colorMaskStabilitySlider_, &QSlider::valueChanged, this,
          updateColorStabilityLabel);

  const auto syncTrackedColorControls = [=]() {
    const bool tracked = trackedColorRadio_->isChecked();
    colorForm->setRowVisible(minObjectRow, tracked);
    colorForm->setRowVisible(persistenceRow, tracked);
    colorForm->setRowVisible(colorStabilityRow, tracked);
  };
  connect(trackedColorRadio_, &QRadioButton::toggled, this,
          [syncTrackedColorControls](bool) { syncTrackedColorControls(); });
  syncTrackedColorControls();

  // Current target preview (click opens QColorDialog)
  {
    auto *swatchRow = new QHBoxLayout;
    pickedHue_ = currentHue;
    pickedSat_ = currentSat;
    pickedVal_ = currentVal;
    colorPickRequested_ = false;

    swatchBtn_ = new QPushButton;
    swatchBtn_->setFixedSize(32, 32);
    swatchBtn_->setCursor(Qt::PointingHandCursor);
    swatchBtn_->setToolTip("Click to pick a key colour manually");
    connect(swatchBtn_, &QPushButton::clicked, this,
            &SettingsDialog::openColorPicker);
    swatchRow->addWidget(swatchBtn_);

    swatchLabel_ = new QLabel;
    swatchLabel_->setWordWrap(true);
    swatchRow->addWidget(swatchLabel_, 1);
    updateSwatchDisplay(hasTarget);
    addFormRow(colorForm, "Target",
               "Current keyed colour. Click the swatch to edit it manually.",
               swatchRow);
  }

  colorLayout->addLayout(colorForm);
  auto *pickFromCamera = new QPushButton("Pick Colour from Camera...");
  pickFromCamera->setToolTip(
      "Close settings and click the target colour in the live camera image.");
  connect(pickFromCamera, &QPushButton::clicked, this, [this]() {
    colorFramePickRequested_ = true;
    accept();
  });
  colorLayout->addWidget(pickFromCamera);
  auto *colorNote = new QLabel(
      "The camera picker gives the best result because it measures the target "
      "under the current lighting. The swatch above is a manual fallback.");
  colorNote->setWordWrap(true);
  colorNote->setForegroundRole(QPalette::PlaceholderText);
  colorLayout->addWidget(colorNote);
  subjectPage->addWidget(colorGroup);
  subjectPage->addStretch(1);

  // ── Lensing Effect group ────────────────────────────────────────────
  auto *lensGroup = new QGroupBox("Lensing Effect");
  auto *lensForm = new QFormLayout(lensGroup);
  configureFormLayout(lensForm);

  strengthSpin_ = makeDoubleSpin(0.0, 10.0, 0.05, 2, {},
      "Controls how strongly the background bends around the lens. Larger "
      "values create a more dramatic effect.");
  strengthSpin_->setValue(static_cast<double>(settings.strength));
  auto *strengthSlider = new QSlider(Qt::Horizontal);
  strengthSlider->setRange(0, 1000);
  strengthSlider->setValue(static_cast<int>(settings.strength * 100.0f));
  strengthSlider->setToolTip(strengthSpin_->toolTip());
  auto *strengthLabel = new QLabel(QString::number(settings.strength, 'f', 1));
  strengthLabel->setMinimumWidth(40);
  auto *strengthRow = new QHBoxLayout;
  strengthRow->addWidget(strengthSlider, 1);
  strengthRow->addWidget(strengthLabel);
  addFormRow(lensForm, "Effect strength", strengthSlider->toolTip(),
             strengthRow);
  connect(strengthSlider, &QSlider::valueChanged, this, [this](int value) {
    strengthSpin_->setValue(static_cast<double>(value) / 100.0);
  });
  connect(strengthSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [strengthSlider, strengthLabel](double value) {
            strengthSlider->setValue(static_cast<int>(std::round(value * 100.0)));
            strengthLabel->setText(QString::number(value, 'f', 1));
          });

  softeningSpin_ = makeDoubleSpin(0.0, 200.0, 1.0, 0, " px",
      "Controls the width and smoothness of the bend around the lens. Larger "
      "values spread the effect over a wider area.");
  softeningSpin_->setValue(static_cast<double>(settings.softening));
  auto *widthSlider = new QSlider(Qt::Horizontal);
  widthSlider->setRange(0, 200);
  widthSlider->setValue(static_cast<int>(settings.softening));
  widthSlider->setToolTip(softeningSpin_->toolTip());
  auto *widthLabel = new QLabel(QString("%1 px").arg(settings.softening, 0, 'f', 0));
  widthLabel->setMinimumWidth(48);
  auto *widthRow = new QHBoxLayout;
  widthRow->addWidget(widthSlider, 1);
  widthRow->addWidget(widthLabel);
  addFormRow(lensForm, "Effect width", widthSlider->toolTip(), widthRow);
  connect(widthSlider, &QSlider::valueChanged, this, [this](int value) {
    softeningSpin_->setValue(static_cast<double>(value));
  });
  connect(softeningSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [widthSlider, widthLabel](double value) {
            widthSlider->setValue(static_cast<int>(std::round(value)));
            widthLabel->setText(QString("%1 px").arg(value, 0, 'f', 0));
          });

  lensEdgeSoftnessSpin_ = makeDoubleSpin(
      0.0, 5.0, 0.1, 1, {},
      "Smooths the detected lens boundary before calculating deflection. "
      "Increase to reduce jagged edges; decrease for a sharper silhouette.");
  lensEdgeSoftnessSpin_->setValue(settings.lensEdgeSoftness);
  auto *edgeSoftnessSlider = new QSlider(Qt::Horizontal);
  edgeSoftnessSlider->setRange(0, 50);
  edgeSoftnessSlider->setValue(
      static_cast<int>(std::round(settings.lensEdgeSoftness * 10.0f)));
  auto *edgeSoftnessRow = new QHBoxLayout;
  edgeSoftnessRow->addWidget(edgeSoftnessSlider, 1);
  edgeSoftnessRow->addWidget(lensEdgeSoftnessSpin_);
  addFormRow(lensForm, "Lens edge softness",
             lensEdgeSoftnessSpin_->toolTip(), edgeSoftnessRow);
  connect(edgeSoftnessSlider, &QSlider::valueChanged, this, [this](int value) {
    lensEdgeSoftnessSpin_->setValue(value / 10.0);
  });
  connect(lensEdgeSoftnessSpin_,
          qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [edgeSoftnessSlider](double value) {
            edgeSoftnessSlider->setValue(
                static_cast<int>(std::round(value * 10.0)));
          });

  padFactorSpin_ = makeIntSpin(1, 10, 1, {},
      "Advanced: extra calculation space used to prevent edge wrap-around. "
      "Increase only if distortion appears on the opposite screen edge.");
  padFactorSpin_->setValue(settings.padFactor);

  distortInsideCheck_ = makeCheck("Keep the lens interior clear",
      "Leave the detected lens area undistorted while bending the background "
      "around it. Turn off to distort the entire detected area.");
  distortInsideCheck_->setChecked(!settings.distortInside);
  addFormRow(lensForm, "Lens appearance", distortInsideCheck_->toolTip(),
              distortInsideCheck_);

  showLensContentsCheck_ = makeCheck(
      "Show the live camera inside the lens",
      "Composite camera pixels inside the detected mask over the lensed "
      "background. This can also be toggled from the View menu.");
  showLensContentsCheck_->setChecked(settings.showLensContents);
  addFormRow(lensForm, "Lens contents", showLensContentsCheck_->toolTip(),
             showLensContentsCheck_);

  effectPage->addWidget(lensGroup);
  effectPage->addStretch(1);

  // ── Backgrounds group ──────────────────────────────────────────────
  auto *bgGroup = new QGroupBox("Backgrounds");
  auto *bgForm = new QFormLayout(bgGroup);
  configureFormLayout(bgForm);

  includedBackgroundsRadio_ = new QRadioButton("Included backgrounds");
  includedBackgroundsRadio_->setToolTip(
      "Use the background collection packaged with GravyLensing.");
  customBackgroundsRadio_ = new QRadioButton("Custom folder");
  customBackgroundsRadio_->setToolTip(
      "Use supported images from a folder on this Mac.");
  auto *backgroundSource = new QVBoxLayout;
  backgroundSource->addWidget(includedBackgroundsRadio_);
  backgroundSource->addWidget(customBackgroundsRadio_);
  addFormRow(bgForm, "Source",
             "Choose the included collection or a folder of your own images.",
             backgroundSource);

  backgroundResolutionCombo_ = new QComboBox;
  backgroundResolutionCombo_->setToolTip(
      "Sets the exact background and FFT geometry. Lower resolutions improve "
      "frame rate and substantially reduce memory use.");
  backgroundResolutionCombo_->addItem("Low - 640 x 360", QSize(640, 360));
  backgroundResolutionCombo_->addItem("480p - 854 x 480", QSize(854, 480));
  backgroundResolutionCombo_->addItem("540p - 960 x 540", QSize(960, 540));
  backgroundResolutionCombo_->addItem("720p - 1280 x 720", QSize(1280, 720));
  backgroundResolutionCombo_->addItem("1080p - 1920 x 1080 (Recommended)",
                                      QSize(1920, 1080));
  backgroundResolutionCombo_->addItem("1440p - 2560 x 1440",
                                      QSize(2560, 1440));
  backgroundResolutionCombo_->addItem("4K - 3840 x 2160",
                                      QSize(3840, 2160));
  backgroundResolutionCombo_->addItem("Custom...", QSize());
  int resolutionIndex = backgroundResolutionCombo_->findData(
      QSize(settings.backgroundWidth, settings.backgroundHeight));
  backgroundResolutionCombo_->setCurrentIndex(
      resolutionIndex >= 0 ? resolutionIndex
                           : backgroundResolutionCombo_->count() - 1);
  addFormRow(bgForm, "Processing size", backgroundResolutionCombo_->toolTip(),
             backgroundResolutionCombo_);

  backgroundWidthSpin_ = makeIntSpin(
      320, 7680, 16, " px", "Exact processed background width.");
  backgroundHeightSpin_ = makeIntSpin(
      180, 4320, 16, " px", "Exact processed background height.");
  backgroundWidthSpin_->setValue(settings.backgroundWidth);
  backgroundHeightSpin_->setValue(settings.backgroundHeight);
  auto *customResolutionRow = new QHBoxLayout;
  customResolutionRow->addWidget(backgroundWidthSpin_);
  customResolutionRow->addWidget(new QLabel("x"));
  customResolutionRow->addWidget(backgroundHeightSpin_);
  customResolutionRow->addStretch(1);
  addFormRow(bgForm, "Custom size",
             "Explicit width and height used for background preprocessing and "
             "lensing calculations.",
             customResolutionRow);
  bgForm->setRowVisible(customResolutionRow, resolutionIndex < 0);
  connect(backgroundResolutionCombo_, &QComboBox::currentIndexChanged, this,
          [this, bgForm, customResolutionRow](int) {
            const QSize size = backgroundResolutionCombo_->currentData().toSize();
            const bool custom = size.isEmpty();
            bgForm->setRowVisible(customResolutionRow, custom);
            if (!custom) {
              backgroundWidthSpin_->setValue(size.width());
              backgroundHeightSpin_->setValue(size.height());
            }
            updateBackgroundStatus();
          });
  connect(backgroundWidthSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int) { updateBackgroundStatus(); });
  connect(backgroundHeightSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [this](int) { updateBackgroundStatus(); });

  backgroundFitCombo_ = new QComboBox;
  backgroundFitCombo_->setToolTip(
      "Controls how images with a different aspect ratio fill the selected "
      "processing size.");
  backgroundFitCombo_->addItem("Crop to Fill - No bars (Recommended)", "crop");
  backgroundFitCombo_->addItem("Fit with Bars - Show whole image", "fit");
  backgroundFitCombo_->addItem("Stretch to Fill - May distort", "stretch");
  const int fitIndex = backgroundFitCombo_->findData(
      QString::fromStdString(settings.backgroundFitMode));
  backgroundFitCombo_->setCurrentIndex(fitIndex >= 0 ? fitIndex : 0);
  addFormRow(bgForm, "Image fitting", backgroundFitCombo_->toolTip(),
             backgroundFitCombo_);
  auto *resolutionGuidance = new QLabel(
      "Resolution controls the amount of work in every frame; frame rate "
      "controls how often that work repeats. If performance drops, lower "
      "resolution first, then lower the camera frame rate. 1080p at 30 fps is "
      "the recommended starting point.");
  resolutionGuidance->setWordWrap(true);
  resolutionGuidance->setForegroundRole(QPalette::PlaceholderText);
  bgForm->addRow({}, resolutionGuidance);

  QHBoxLayout *backgroundFolderRow = nullptr;
  {
    backgroundFolderRow = new QHBoxLayout;
    backgroundsDirEdit_ =
        new QLineEdit(QString::fromStdString(settings.backgroundsDir));
    backgroundsDirEdit_->setSizePolicy(QSizePolicy::Expanding,
                                       QSizePolicy::Fixed);
    backgroundsDirEdit_->setClearButtonEnabled(true);
    backgroundsDirEdit_->setToolTip(backgroundsDirEdit_->text());
    backgroundsDirEdit_->setPlaceholderText("Path to backgrounds folder...");
    browseBgBtn_ = new QPushButton("Browse...");
    browseBgBtn_->setToolTip(
        "Select a folder containing background images.");
    backgroundFolderRow->addWidget(backgroundsDirEdit_, 1);
    backgroundFolderRow->addWidget(browseBgBtn_);
    addFormRow(bgForm, "Custom folder",
                "Folder scanned for background images. Use Browse to choose "
                "your own collection; Restore Defaults returns to built-ins.",
                backgroundFolderRow);
    connect(browseBgBtn_, &QPushButton::clicked, this, [this]() {
      const QString dir = QFileDialog::getExistingDirectory(
          this, "Select Backgrounds Folder", backgroundsDirEdit_->text());
      if (!dir.isEmpty()) {
        customBackgroundsRadio_->setChecked(true);
        backgroundsDirEdit_->setText(dir);
        backgroundsDirEdit_->setToolTip(dir);
        const size_t count =
            Backgrounds::discoverableImageCount(dir.toStdString());
        const QString formats =
            "Images do not need to be TIFFs. Supported formats are PNG, JPEG, "
            "BMP, GIF, TIFF, WebP, and SVG when a decoder is available.";
        const QString resolution =
            QString("Images will be preprocessed and cached at %1 x %2. Higher "
                    "resolution and frame-rate targets require substantially "
                    "more processing power.")
                .arg(backgroundWidthSpin_->value())
                .arg(backgroundHeightSpin_->value());
        if (count == 0) {
          QMessageBox::warning(
              this, "No Usable Backgrounds",
              "No readable background images were found in this folder.\n\n" +
                  formats + "\n\n" + resolution);
        } else {
          QMessageBox::information(
              this, "Background Folder Ready",
              QString("Found %1 usable background%2.\n\n%3\n\n%4")
                  .arg(count)
                  .arg(count == 1 ? "" : "s")
                  .arg(formats)
                  .arg(resolution));
        }
      }
    });
    connect(backgroundsDirEdit_, &QLineEdit::textChanged, this,
            [this](const QString &t) {
              backgroundsDirEdit_->setToolTip(t);
              updateBackgroundStatus();
            });
  }

  backgroundStatus_ = new QLabel;
  backgroundStatus_->setWordWrap(true);
  backgroundStatus_->setForegroundRole(QPalette::PlaceholderText);
  bgForm->addRow({}, backgroundStatus_);
  auto *cacheStatus = new QLabel;
  cacheStatus->setForegroundRole(QPalette::PlaceholderText);
  const auto updateCacheStatus = [cacheStatus]() {
    cacheStatus->setText(
        QString("%1 processed image%2, %3 MB on disk")
            .arg(Backgrounds::cacheImageCount())
            .arg(Backgrounds::cacheImageCount() == 1 ? "" : "s")
            .arg(Backgrounds::cacheSizeBytes() / (1024.0 * 1024.0), 0, 'f', 1));
  };
  updateCacheStatus();
  auto *rebuildCache = new QPushButton("Rebuild Cache");
  rebuildCache->setToolTip(
      "Clear processed backgrounds and rebuild the selected source at the "
      "current resolution and fitting mode.");
  auto *cacheRow = new QHBoxLayout;
  cacheRow->addWidget(cacheStatus, 1);
  cacheRow->addWidget(rebuildCache);
  addFormRow(bgForm, "Cache", rebuildCache->toolTip(), cacheRow);
  connect(rebuildCache, &QPushButton::clicked, this, [=]() {
    backgroundCacheRebuildRequested_ = true;
    rebuildCache->setText("Rebuild on Save");
    rebuildCache->setEnabled(false);
  });
  const bool includedBackgrounds =
      settings.backgroundsDir == AppSettings().backgroundsDir;
  includedBackgroundsRadio_->setChecked(includedBackgrounds);
  customBackgroundsRadio_->setChecked(!includedBackgrounds);
  const auto syncBackgroundSource = [=]() {
    bgForm->setRowVisible(backgroundFolderRow,
                          customBackgroundsRadio_->isChecked());
    updateBackgroundStatus();
  };
  connect(customBackgroundsRadio_, &QRadioButton::toggled, this,
          [syncBackgroundSource](bool) { syncBackgroundSource(); });
  syncBackgroundSource();

  autoCycleCheck_ = new QCheckBox("Auto-cycle backgrounds");
  autoCycleCheck_->setToolTip(
      "When enabled, the background changes automatically after the "
      "specified interval. When disabled, switch with the arrow keys.");
  autoCycleCheck_->setChecked(settings.secondsPerBackground > 0);
  addFormRow(bgForm, "Auto-cycle", autoCycleCheck_->toolTip(),
             autoCycleCheck_);

  secondsPerBackgroundSpin_ = new QSpinBox;
  secondsPerBackgroundSpin_->setRange(1, 3600);
  secondsPerBackgroundSpin_->setSingleStep(5);
  secondsPerBackgroundSpin_->setSuffix(" s");
  secondsPerBackgroundSpin_->setToolTip(
      "How long each background remains visible before moving to the next.");
  secondsPerBackgroundSpin_->setValue(
      std::max(1, settings.secondsPerBackground));
  secondsPerBackgroundSpin_->setEnabled(settings.secondsPerBackground > 0);
  connect(autoCycleCheck_, &QCheckBox::toggled, secondsPerBackgroundSpin_,
          &QSpinBox::setEnabled);
  addFormRow(bgForm, "Cycle interval", secondsPerBackgroundSpin_->toolTip(),
             secondsPerBackgroundSpin_);

  backgroundsPage->addWidget(bgGroup);
  backgroundsPage->addStretch(1);

  // ── Performance group ───────────────────────────────────────────────
  auto *perfGroup = new QGroupBox("Advanced");
  auto *perfForm = new QFormLayout(perfGroup);
  configureFormLayout(perfForm);

  nthreadsSpin_ = makeIntSpin(
      1, 256, 1, {},
      QString("Worker threads used for lensing. This Mac reports %1 logical "
              "cores; the default leaves two available for the camera, UI, "
              "and system.")
          .arg(QThread::idealThreadCount()));
  nthreadsSpin_->setValue(settings.nthreads);
  addFormRow(perfForm, "Worker threads", nthreadsSpin_->toolTip(),
             nthreadsSpin_);

  lowerResSpin_ = makeDoubleSpin(
      0.1, 1.0, 0.05, 2, {},
      "Fraction of the cached background resolution used for masks, FFTs, and "
      "lensing calculations. The result is upscaled to the exact background "
      "size. Normally controlled by the Quality preset.");
  lowerResSpin_->setValue(settings.lowerRes);
  auto *lowerResSlider = new QSlider(Qt::Horizontal);
  lowerResSlider->setRange(10, 100);
  lowerResSlider->setSingleStep(5);
  lowerResSlider->setValue(
      static_cast<int>(std::round(settings.lowerRes * 100.0f)));
  auto *lowerResRow = new QHBoxLayout;
  lowerResRow->addWidget(lowerResSlider, 1);
  lowerResRow->addWidget(lowerResSpin_);
  addFormRow(perfForm, "Calculation scale", lowerResSpin_->toolTip(),
             lowerResRow);
  connect(lowerResSlider, &QSlider::valueChanged, this, [this](int value) {
    lowerResSpin_->setValue(value / 100.0);
  });
  connect(lowerResSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
          [lowerResSlider, syncQualityModeFromControls](double value) {
            lowerResSlider->setValue(
                static_cast<int>(std::round(value * 100.0)));
            syncQualityModeFromControls();
          });

  auto *paddingCombo = new QComboBox;
  paddingCombo->setToolTip(
      "Extra calculation space that prevents distortion wrapping around screen "
      "edges. Standard is recommended; larger values use much more memory.");
  paddingCombo->addItem("Standard (Recommended)", 2);
  paddingCombo->addItem("Extra", 3);
  paddingCombo->addItem("Maximum", 4);
  paddingCombo->addItem("Custom...", -1);
  const int paddingPreset = paddingCombo->findData(settings.padFactor);
  paddingCombo->setCurrentIndex(
      paddingPreset >= 0 ? paddingPreset : paddingCombo->count() - 1);
  padFactorSpin_->setVisible(paddingPreset < 0);
  auto *paddingRow = new QHBoxLayout;
  paddingRow->addWidget(paddingCombo, 1);
  paddingRow->addWidget(padFactorSpin_);
  addFormRow(perfForm, "Edge protection", paddingCombo->toolTip(), paddingRow);
  connect(paddingCombo, &QComboBox::currentIndexChanged, this,
          [this, paddingCombo](int) {
            const int factor = paddingCombo->currentData().toInt();
            padFactorSpin_->setVisible(factor < 0);
            if (factor > 0)
              padFactorSpin_->setValue(factor);
          });
  connect(padFactorSpin_, qOverload<int>(&QSpinBox::valueChanged), this,
          [paddingCombo](int value) {
            const int preset = paddingCombo->findData(value);
            QSignalBlocker blocker(paddingCombo);
            paddingCombo->setCurrentIndex(
                preset >= 0 ? preset : paddingCombo->count() - 1);
          });
  const auto syncCustomQualityControls = [=]() {
    const bool custom =
        qualityModeCombo_->currentData().toString() == QLatin1String("custom");
    personForm->setRowVisible(visionSizeRow, custom);
    personForm->setRowVisible(stabilizationRow, custom);
    perfForm->setRowVisible(lowerResRow, custom);
  };
  connect(qualityModeCombo_, &QComboBox::currentIndexChanged, this,
          [syncCustomQualityControls](int) { syncCustomQualityControls(); });
  syncCustomQualityControls();

  debugGridCheck_ = makeCheck("Show diagnostic grid",
      "Display the 2×2 debug view (camera, mask, background, lensed) "
      "instead of the final effect. Useful for setup and troubleshooting.");
  debugGridCheck_->setChecked(settings.debugGrid);
  addFormRow(perfForm, "Debug grid", debugGridCheck_->toolTip(),
             debugGridCheck_);

  advancedPage->addWidget(perfGroup);

  // ── Note about restart ──────────────────────────────────────────────
  auto *note = new QLabel(
      "Tip: start with Person Detection and Balanced quality. During a session, "
      "use the arrow keys to change backgrounds and Cmd+, to return here.");
  note->setWordWrap(true);
  note->setStyleSheet(
      "QLabel { color: #888; font-size: 11px; padding-top: 6px; }");
  advancedPage->addWidget(note);
  advancedPage->addStretch(1);

  const auto syncModeGroups = [=]() {
    const bool colorMode = colorDetectionRadio_->isChecked();
    personGroup->setVisible(!colorMode);
    colorGroup->setVisible(colorMode);
  };
  connect(colorDetectionRadio_, &QRadioButton::toggled, this,
          [=](bool) { syncModeGroups(); });
  syncModeGroups();

  // ── Button box ──────────────────────────────────────────────────────
  auto *buttons = new QDialogButtonBox(
      QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok |
      QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Ok)->setText(acceptLabel);
  buttons->button(QDialogButtonBox::Ok)->setDefault(true);
  buttons->button(QDialogButtonBox::Ok)->setToolTip(
      "Apply these settings and start the camera session.");
  buttons->button(QDialogButtonBox::Cancel)->setToolTip(
      "Close without applying these changes.");
  buttons->button(QDialogButtonBox::RestoreDefaults)->setToolTip(
      "Reset every setting to the recommended defaults.");
  mainLayout->addWidget(buttons);

  connect(buttons->button(QDialogButtonBox::RestoreDefaults),
          &QPushButton::clicked, this,
          [this, syncModeGroups, paddingCombo]() {
            const AppSettings defaults;
#ifdef __APPLE__
            { const int idx = cameraCombo_->findData(defaults.deviceIndex); if (idx >= 0) cameraCombo_->setCurrentIndex(idx); }
#else
            deviceIndexSpin_->setValue(defaults.deviceIndex);
#endif
            { const int idx = fpsCombo_->findData(defaults.fps); if (idx >= 0) fpsCombo_->setCurrentIndex(idx); }
            { const int idx = cameraResolutionCombo_->findData(QSize(defaults.cameraWidth, defaults.cameraHeight)); if (idx >= 0) cameraResolutionCombo_->setCurrentIndex(idx); }
            flipCheck_->setChecked(defaults.flip);
            selectROICheck_->setChecked(defaults.selectROI);
            colorDetectionRadio_->setChecked(defaults.maskMode == "color");
            personDetectionRadio_->setChecked(defaults.maskMode != "color");
            trackedColorRadio_->setChecked(
                defaults.colorModeType == "tracked_blob");
            fixedColorRadio_->setChecked(
                defaults.colorModeType != "tracked_blob");
            colorHueTolSpin_->setValue(defaults.colorHueTol);
            colorSatTolSpin_->setValue(defaults.colorSatTol);
            colorValTolSpin_->setValue(defaults.colorValTol);
            colorMinObjectAreaSpin_->setValue(defaults.colorMinObjectArea);
            colorPersistenceSpin_->setValue(defaults.colorPersistenceFrames);
            colorMaskStabilitySlider_->setValue(static_cast<int>(
                std::round((1.0f - defaults.colorMaskSmooth) * 100.0f)));
            visionSizeSpin_->setValue(defaults.visionSize);
            { const int idx = qualityModeCombo_->findData(QString::fromStdString(defaults.qualityMode)); if (idx >= 0) qualityModeCombo_->setCurrentIndex(idx); }
            temporalSmoothSpin_->setValue(
                static_cast<double>(defaults.temporalSmooth));
            lowerResSpin_->setValue(defaults.lowerRes);
            personSensitivitySlider_->setValue(defaults.personSensitivity);
            strengthSpin_->setValue(
                static_cast<double>(defaults.strength));
            softeningSpin_->setValue(
                static_cast<double>(defaults.softening));
            lensEdgeSoftnessSpin_->setValue(defaults.lensEdgeSoftness);
            padFactorSpin_->setValue(defaults.padFactor);
            distortInsideCheck_->setChecked(!defaults.distortInside);
            showLensContentsCheck_->setChecked(defaults.showLensContents);
            nthreadsSpin_->setValue(defaults.nthreads);
            backgroundsDirEdit_->setText(
                QString::fromStdString(defaults.backgroundsDir));
            includedBackgroundsRadio_->setChecked(true);
            backgroundWidthSpin_->setValue(defaults.backgroundWidth);
            backgroundHeightSpin_->setValue(defaults.backgroundHeight);
            { const int idx = backgroundResolutionCombo_->findData(QSize(defaults.backgroundWidth, defaults.backgroundHeight)); if (idx >= 0) backgroundResolutionCombo_->setCurrentIndex(idx); }
            { const int idx = backgroundFitCombo_->findData(QString::fromStdString(defaults.backgroundFitMode)); if (idx >= 0) backgroundFitCombo_->setCurrentIndex(idx); }
            autoCycleCheck_->setChecked(defaults.secondsPerBackground > 0);
            secondsPerBackgroundSpin_->setValue(
                std::max(1, defaults.secondsPerBackground));

            // Reset ROI and colour-pick state
            hasROI_ = false;
            roiX_ = roiY_ = roiW_ = roiH_ = 0;
            roiInfoLabel_->setText("No region selected — full frame in use.");
            colorPickRequested_ = false;
             pickedHue_ = pickedSat_ = pickedVal_ = 0;
             updateSwatchDisplay(false);
             debugGridCheck_->setChecked(defaults.debugGrid);
             syncModeGroups();
           });

  connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

AppSettings SettingsDialog::settings() const {
  AppSettings s;
  s.nthreads = nthreadsSpin_->value();
#ifdef __APPLE__
  s.deviceIndex = cameraCombo_->currentData().toInt();
#else
  s.deviceIndex = deviceIndexSpin_->value();
#endif
  s.fps = fpsCombo_->currentData().toInt();
  const QSize cameraSize = cameraResolutionCombo_->currentData().toSize();
  s.cameraWidth = cameraSize.width();
  s.cameraHeight = cameraSize.height();
  s.flip = flipCheck_->isChecked();
  s.selectROI = selectROICheck_->isChecked();
  s.maskMode = colorDetectionRadio_->isChecked() ? "color" : "person";
  s.colorModeType =
      trackedColorRadio_->isChecked() ? "tracked_blob" : "fixed_key";
  s.colorHueTol = colorHueTolSpin_->value();
  s.colorSatTol = colorSatTolSpin_->value();
  s.colorValTol = colorValTolSpin_->value();
  s.colorMinObjectArea = colorMinObjectAreaSpin_->value();
  s.colorPersistenceFrames = colorPersistenceSpin_->value();
  s.colorMaskSmooth =
      1.0f - colorMaskStabilitySlider_->value() / 100.0f;
  s.visionSize = visionSizeSpin_->value();
  s.qualityMode = qualityModeCombo_->currentData().toString().toStdString();
  s.strength = static_cast<float>(strengthSpin_->value());
  s.softening = static_cast<float>(softeningSpin_->value());
  s.lensEdgeSoftness =
      static_cast<float>(lensEdgeSoftnessSpin_->value());
  s.padFactor = padFactorSpin_->value();
  s.distortInside = !distortInsideCheck_->isChecked();
  s.showLensContents = showLensContentsCheck_->isChecked();
  s.temporalSmooth = static_cast<float>(temporalSmoothSpin_->value());
  s.lowerRes = static_cast<float>(lowerResSpin_->value());
  s.personSensitivity = personSensitivitySlider_->value();
  s.backgroundsDir = includedBackgroundsRadio_->isChecked()
                         ? AppSettings().backgroundsDir
                         : backgroundsDirEdit_->text().toStdString();
  s.backgroundWidth = backgroundWidthSpin_->value();
  s.backgroundHeight = backgroundHeightSpin_->value();
  s.backgroundFitMode =
      backgroundFitCombo_->currentData().toString().toStdString();
  s.rebuildBackgroundCache = backgroundCacheRebuildRequested_;
  s.secondsPerBackground =
      autoCycleCheck_->isChecked() ? std::max(1, secondsPerBackgroundSpin_->value()) : -1;
  s.debugGrid = debugGridCheck_->isChecked();
  return s;
}

void SettingsDialog::updateSwatchDisplay(bool hasTarget) {
  if (hasTarget) {
    const int hDeg = static_cast<int>(pickedHue_ * 2.0f);
    const int sPct = static_cast<int>(pickedSat_ * 100.0f / 255.0f);
    const int vPct = static_cast<int>(pickedVal_ * 100.0f / 255.0f);
    swatchBtn_->setStyleSheet(
        QString("background-color: hsv(%1,%2%,%3%); "
                "border: 2px solid #888; border-radius: 6px;")
            .arg(hDeg)
            .arg(sPct)
            .arg(vPct));
    swatchLabel_->setText(
        QString("H=%1\u00B0  S=%2  V=%3\n(click to change, or use Shift+S / "
                "Select Color from Frame to pick from the camera)")
            .arg(pickedHue_, 0, 'f', 1)
            .arg(pickedSat_, 0, 'f', 1)
            .arg(pickedVal_, 0, 'f', 1));
  } else {
    swatchBtn_->setStyleSheet(
        "background-color: #555; border: 2px solid #888; "
        "border-radius: 6px;");
    swatchLabel_->setText("No colour selected — click swatch to pick manually,\n"
                          "or use Shift+S / Select Color from Frame to pick from camera");
  }
}

void SettingsDialog::updateBackgroundStatus() {
  const std::string dir = includedBackgroundsRadio_->isChecked()
                              ? AppSettings().backgroundsDir
                              : backgroundsDirEdit_->text().toStdString();
  const size_t count = Backgrounds::discoverableImageCount(dir);
  const QString sizeNote = QString("; cached at %1 x %2")
                               .arg(backgroundWidthSpin_->value())
                               .arg(backgroundHeightSpin_->value());
  backgroundStatus_->setText(count > 0
                                 ? QString("%1 supported image%2 found%3")
                                       .arg(count)
                                       .arg(count == 1 ? "" : "s")
                                       .arg(sizeNote)
                                 : "No supported images found in this location");
}

void SettingsDialog::openColorPicker() {
  // Convert current HSV → QColor
  const int hDeg = static_cast<int>(pickedHue_ * 2.0f);
  const int s255 = static_cast<int>(pickedSat_);
  const int v255 = static_cast<int>(pickedVal_);
  QColor initial = (s255 > 0 || v255 > 0)
                       ? QColor::fromHsv(hDeg, s255, v255)
                       : QColor(Qt::red);

  QColor chosen = QColorDialog::getColor(initial, this, "Pick Key Colour",
                                         QColorDialog::DontUseNativeDialog);
  if (!chosen.isValid())
    return;

  // QColor HSV → OpenCV HSV (H is 0-359, OpenCV is 0-180)
  pickedHue_ = chosen.hueF() * 180.0f;
  pickedSat_ = chosen.saturationF() * 255.0f;
  pickedVal_ = chosen.valueF() * 255.0f;
  colorPickRequested_ = true;

  updateSwatchDisplay(true);
}

#ifdef __APPLE__
void SettingsDialog::refreshCameras() {
  const QString selected = cameraCombo_->currentText();
  const int selectedIndex = cameraCombo_->currentData().toInt();
  const auto names = AvFoundationCamera::availableDeviceNames();
  if (static_cast<size_t>(cameraCombo_->count()) == names.size()) {
    bool unchanged = true;
    for (int i = 0; i < cameraCombo_->count(); ++i)
      unchanged &= cameraCombo_->itemText(i) == QString::fromStdString(names[i]);
    if (unchanged)
      return;
  }

  QSignalBlocker blocker(cameraCombo_);
  cameraCombo_->clear();
  for (size_t i = 0; i < names.size(); ++i)
    cameraCombo_->addItem(QString::fromStdString(names[i]), static_cast<int>(i));
  if (cameraCombo_->count() == 0) {
    cameraCombo_->addItem("No cameras found", 0);
    cameraCombo_->setEnabled(false);
    return;
  }
  cameraCombo_->setEnabled(true);
  int replacement = cameraCombo_->findText(selected);
  if (replacement < 0)
    replacement = cameraCombo_->findData(selectedIndex);
  cameraCombo_->setCurrentIndex(replacement >= 0 ? replacement : 0);
}
#endif
