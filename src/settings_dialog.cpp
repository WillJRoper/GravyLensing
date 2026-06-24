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
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSizePolicy>
#include <QVBoxLayout>

static QDoubleSpinBox *makeDoubleSpin(double min, double max, double step,
                                       int decimals, const QString &suffix,
                                       const QString &tooltip) {
  auto *s = new QDoubleSpinBox;
  s->setRange(min, max);
  s->setSingleStep(step);
  s->setDecimals(decimals);
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
  setMinimumWidth(1050);
  setSizeGripEnabled(true);

  auto *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(20, 20, 20, 14);

  auto *columns = new QHBoxLayout;
  columns->setSpacing(24);
  mainLayout->addLayout(columns);

  auto *leftColumn = new QVBoxLayout;
  auto *rightColumn = new QVBoxLayout;
  columns->addLayout(leftColumn, 1);
  columns->addLayout(rightColumn, 1);

  // ── Camera group ────────────────────────────────────────────────────
  auto *cameraGroup = new QGroupBox("Camera");
  auto *cameraForm = new QFormLayout(cameraGroup);
  cameraForm->setLabelAlignment(Qt::AlignRight);

  deviceIndexSpin_ = makeIntSpin(0, 99, 1, {},
      "Which camera device to open (0 is the built-in webcam).");
  deviceIndexSpin_->setValue(settings.deviceIndex);
  cameraForm->addRow("Device index", deviceIndexSpin_);

  flipCheck_ = makeCheck("Mirror camera feed horizontally",
      "Flip the image so movement in the real world and on-screen are "
      "directionally consistent.");
  flipCheck_->setChecked(settings.flip);
  cameraForm->addRow("Flip", flipCheck_);

  selectROICheck_ = makeCheck("Show region selector at startup",
      "On the next pipeline start, open an interactive ROI selection "
      "window.");
  selectROICheck_->setChecked(settings.selectROI);
  cameraForm->addRow("Region of interest", selectROICheck_);

  leftColumn->addWidget(cameraGroup);

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
  auto *selectRoiBtn = new QPushButton("Select Region...");
  selectRoiBtn->setToolTip(
      "Open the interactive region selector on the camera feed.");
  connect(selectRoiBtn, &QPushButton::clicked, this, [this]() {
    roiSelectRequested_ = true;
    accept();
  });
  roiBtnRow->addWidget(selectRoiBtn);

  auto *clearRoiBtn = new QPushButton("Clear");
  clearRoiBtn->setToolTip("Remove the current region, reverting to full frame.");
  clearRoiBtn->setEnabled(hasROI_);
  connect(clearRoiBtn, &QPushButton::clicked, this, [this]() {
    hasROI_ = false;
    roiX_ = roiY_ = roiW_ = roiH_ = 0;
    roiInfoLabel_->setText("No region selected — full frame in use.");
    auto *btn = qobject_cast<QPushButton *>(sender());
    if (btn) btn->setEnabled(false);
  });
  roiBtnRow->addWidget(clearRoiBtn);
  roiLayout->addLayout(roiBtnRow);
  leftColumn->addWidget(roiGroup);

  // ── Mode (mask type selection) ─────────────────────────────────────
  auto *modeGroup = new QGroupBox("Mode");
  auto *modeForm = new QFormLayout(modeGroup);
  modeForm->setLabelAlignment(Qt::AlignRight);

  maskModeCombo_ = new QComboBox;
  maskModeCombo_->setToolTip(
      "Person uses an AI segmentation model; Color tracks a user-selected "
      "HSV colour.");
  maskModeCombo_->addItem("Person (AI segmentation)", "person");
  maskModeCombo_->addItem("Color tracking", "color");
  if (settings.maskMode == "color")
    maskModeCombo_->setCurrentIndex(1);
  modeForm->addRow("Mask mode", maskModeCombo_);
  leftColumn->addWidget(modeGroup);

  // ── Person Detection group ──────────────────────────────────────────
  auto *personGroup = new QGroupBox("Person Detection");
  auto *personForm = new QFormLayout(personGroup);
  personForm->setLabelAlignment(Qt::AlignRight);

  {
    auto *row = new QHBoxLayout;
    modelPathEdit_ = new QLineEdit(QString::fromStdString(settings.modelPath));
    modelPathEdit_->setSizePolicy(QSizePolicy::Expanding,
                                  QSizePolicy::Fixed);
    modelPathEdit_->setClearButtonEnabled(true);
    modelPathEdit_->setCursorPosition(modelPathEdit_->text().size());
    modelPathEdit_->setToolTip(modelPathEdit_->text());
    modelPathEdit_->setPlaceholderText("Path to TorchScript model...");
    browseBtn_ = new QPushButton("Browse...");
    browseBtn_->setToolTip("Open a file dialog to locate the segmentation model.");
    row->addWidget(modelPathEdit_, 1);
    row->addWidget(browseBtn_);
    personForm->addRow("Model path", row);
    connect(browseBtn_, &QPushButton::clicked, this,
            &SettingsDialog::browseModelPath);
    connect(modelPathEdit_, &QLineEdit::textChanged, this,
            [this](const QString &text) { modelPathEdit_->setToolTip(text); });
  }

  modelSizeSpin_ = makeIntSpin(128, 1024, 128, " px",
      "Larger models capture finer mask detail but run slower.");
  modelSizeSpin_->setValue(settings.modelSize);
  personForm->addRow("Model size", modelSizeSpin_);

  temporalSmoothSpin_ = makeDoubleSpin(0.0, 1.0, 0.05, 2, {},
      "Higher values blend the mask more heavily with the previous frame, "
      "reducing flicker at the cost of responsiveness.");
  temporalSmoothSpin_->setValue(static_cast<double>(settings.temporalSmooth));
  personForm->addRow("Temporal smooth", temporalSmoothSpin_);

  leftColumn->addWidget(personGroup);

  // ── Color Detection (subtype + tolerances) ─────────────────────────
  auto *colorGroup = new QGroupBox("Color Detection");
  auto *colorLayout = new QVBoxLayout(colorGroup);
  auto *colorForm = new QFormLayout;
  colorForm->setLabelAlignment(Qt::AlignRight);

  colorModeTypeCombo_ = new QComboBox;
  colorModeTypeCombo_->setToolTip(
      "Fixed Color Key thresholds the target HSV range directly. "
      "Tracked Color Blob adds blob-tracking heuristics.");
  colorModeTypeCombo_->addItem("Fixed Color Key", "fixed_key");
  colorModeTypeCombo_->addItem("Tracked Color Blob", "tracked_blob");
  if (settings.colorModeType == "tracked_blob")
    colorModeTypeCombo_->setCurrentIndex(1);
  colorForm->addRow("Mode", colorModeTypeCombo_);

  colorHueTolSpin_ = makeIntSpin(1, 90, 1, {},
      "Wider values include more hues around the target. "
      "Hue wraps at 0/180.");
  colorHueTolSpin_->setValue(settings.colorHueTol);
  colorForm->addRow("Hue tolerance", colorHueTolSpin_);

  colorSatTolSpin_ = makeIntSpin(1, 255, 5, {},
      "Wider values include more saturation variation.");
  colorSatTolSpin_->setValue(settings.colorSatTol);
  colorForm->addRow("Saturation tolerance", colorSatTolSpin_);

  colorValTolSpin_ = makeIntSpin(1, 255, 5, {},
      "Wider values include more brightness variation.");
  colorValTolSpin_->setValue(settings.colorValTol);
  colorForm->addRow("Value tolerance", colorValTolSpin_);

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
    colorForm->addRow("Target", swatchRow);
  }

  colorLayout->addLayout(colorForm);
  auto *colorNote = new QLabel(
      "Choose or re-choose the keyed colour with Shift+S or "
      "File > Select Color...");
  colorNote->setWordWrap(true);
  colorLayout->addWidget(colorNote);
  leftColumn->addWidget(colorGroup);
  leftColumn->addStretch(1);

  // ── Lensing Effect group ────────────────────────────────────────────
  auto *lensGroup = new QGroupBox("Lensing Effect");
  auto *lensForm = new QFormLayout(lensGroup);
  lensForm->setLabelAlignment(Qt::AlignRight);

  strengthSpin_ = makeDoubleSpin(0.0, 10.0, 0.01, 3, {},
      "Strength multiplier applied to the deflection field. Larger "
      "values produce more dramatic lensing.");
  strengthSpin_->setValue(static_cast<double>(settings.strength));
  lensForm->addRow("Strength", strengthSpin_);

  softeningSpin_ = makeDoubleSpin(0.0, 200.0, 1.0, 1, " px",
      "Softens the deflection kernel so nearby background pixels are "
      "affected more smoothly.");
  softeningSpin_->setValue(static_cast<double>(settings.softening));
  lensForm->addRow("Softening radius", softeningSpin_);

  padFactorSpin_ = makeIntSpin(1, 10, 1, {},
      "FFT padding multiplier. Larger values reduce wrap-around "
      "artifacts but use more memory.");
  padFactorSpin_->setValue(settings.padFactor);
  lensForm->addRow("FFT pad factor", padFactorSpin_);

  lowerResSpin_ = makeDoubleSpin(0.1, 1.0, 0.1, 2, {},
      "Fraction of the background resolution at which lensing is "
      "computed. 1.0 = full resolution; 0.5 = half resolution (faster).");
  lowerResSpin_->setValue(static_cast<double>(settings.lowerRes));
  lensForm->addRow("Resolution scale", lowerResSpin_);

  distortInsideCheck_ = makeCheck("Distort inside the mask",
      "When enabled, the interior of the mask is also lensed (not only "
      "the background around it).");
  distortInsideCheck_->setChecked(settings.distortInside);
  lensForm->addRow("Inside mask", distortInsideCheck_);

  rightColumn->addWidget(lensGroup);

  // ── Backgrounds group ──────────────────────────────────────────────
  auto *bgGroup = new QGroupBox("Backgrounds");
  auto *bgForm = new QFormLayout(bgGroup);
  bgForm->setLabelAlignment(Qt::AlignRight);

  {
    auto *row = new QHBoxLayout;
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
    row->addWidget(backgroundsDirEdit_, 1);
    row->addWidget(browseBgBtn_);
    bgForm->addRow("Directory", row);
    connect(browseBgBtn_, &QPushButton::clicked, this, [this]() {
      const QString dir = QFileDialog::getExistingDirectory(
          this, "Select Backgrounds Folder", backgroundsDirEdit_->text());
      if (!dir.isEmpty()) {
        backgroundsDirEdit_->setText(dir);
        backgroundsDirEdit_->setToolTip(dir);
      }
    });
    connect(backgroundsDirEdit_, &QLineEdit::textChanged, this,
            [this](const QString &t) { backgroundsDirEdit_->setToolTip(t); });
  }

  autoCycleCheck_ = new QCheckBox("Auto-cycle backgrounds");
  autoCycleCheck_->setToolTip(
      "When enabled, the background changes automatically after the "
      "specified interval. When disabled, switch manually with 0–9 keys.");
  autoCycleCheck_->setChecked(settings.secondsPerBackground > 0);
  bgForm->addRow("Auto-cycle", autoCycleCheck_);

  secondsPerBackgroundSpin_ = new QSpinBox;
  secondsPerBackgroundSpin_->setRange(1, 3600);
  secondsPerBackgroundSpin_->setSingleStep(5);
  secondsPerBackgroundSpin_->setSuffix(" s");
  secondsPerBackgroundSpin_->setToolTip(
      "Seconds between automatic background changes.");
  secondsPerBackgroundSpin_->setValue(
      std::max(1, settings.secondsPerBackground));
  secondsPerBackgroundSpin_->setEnabled(settings.secondsPerBackground > 0);
  connect(autoCycleCheck_, &QCheckBox::toggled, secondsPerBackgroundSpin_,
          &QSpinBox::setEnabled);
  bgForm->addRow("Cycle interval", secondsPerBackgroundSpin_);

  rightColumn->addWidget(bgGroup);

  // ── Performance group ───────────────────────────────────────────────
  auto *perfGroup = new QGroupBox("Performance");
  auto *perfForm = new QFormLayout(perfGroup);
  perfForm->setLabelAlignment(Qt::AlignRight);

  nthreadsSpin_ = makeIntSpin(2, 256, 1, {},
      "Number of CPU worker threads (Qt reserves 3; the remainder are "
      "used for FFT and segmentation work).");
  nthreadsSpin_->setValue(settings.nthreads);
  perfForm->addRow("CPU threads", nthreadsSpin_);

  debugGridCheck_ = makeCheck("Show diagnostic grid",
      "Display the 2×2 debug view (camera, mask, background, lensed) "
      "instead of the lensed-only view.");
  debugGridCheck_->setChecked(settings.debugGrid);
  perfForm->addRow("Debug grid", debugGridCheck_);

  rightColumn->addWidget(perfGroup);

  // ── Note about restart ──────────────────────────────────────────────
  auto *note = new QLabel(
      "Session settings are applied when you start or restart the session.\n"
      "Color and region selection are manual actions from the File menu.");
  note->setWordWrap(true);
  note->setStyleSheet(
      "QLabel { color: #888; font-size: 11px; padding-top: 6px; }");
  rightColumn->addWidget(note);
  rightColumn->addStretch(1);

  // ── Button box ──────────────────────────────────────────────────────
  auto *buttons = new QDialogButtonBox(
      QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok |
      QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Ok)->setText(acceptLabel);
  mainLayout->addWidget(buttons);

  connect(buttons->button(QDialogButtonBox::RestoreDefaults),
          &QPushButton::clicked, this, [this]() {
            const AppSettings defaults;
            deviceIndexSpin_->setValue(defaults.deviceIndex);
            flipCheck_->setChecked(defaults.flip);
            selectROICheck_->setChecked(defaults.selectROI);
            maskModeCombo_->setCurrentIndex(
                defaults.maskMode == "color" ? 1 : 0);
            colorModeTypeCombo_->setCurrentIndex(
                defaults.colorModeType == "tracked_blob" ? 1 : 0);
            colorHueTolSpin_->setValue(defaults.colorHueTol);
            colorSatTolSpin_->setValue(defaults.colorSatTol);
            colorValTolSpin_->setValue(defaults.colorValTol);
            modelPathEdit_->setText(
                QString::fromStdString(defaults.modelPath));
            modelSizeSpin_->setValue(defaults.modelSize);
            temporalSmoothSpin_->setValue(
                static_cast<double>(defaults.temporalSmooth));
            strengthSpin_->setValue(
                static_cast<double>(defaults.strength));
            softeningSpin_->setValue(
                static_cast<double>(defaults.softening));
            padFactorSpin_->setValue(defaults.padFactor);
            lowerResSpin_->setValue(
                static_cast<double>(defaults.lowerRes));
            distortInsideCheck_->setChecked(defaults.distortInside);
            nthreadsSpin_->setValue(defaults.nthreads);
            backgroundsDirEdit_->setText(
                QString::fromStdString(defaults.backgroundsDir));
            autoCycleCheck_->setChecked(defaults.secondsPerBackground > 0);
            secondsPerBackgroundSpin_->setValue(
                std::max(1, defaults.secondsPerBackground));
            debugGridCheck_->setChecked(defaults.debugGrid);
            modelPathEdit_->setCursorPosition(
                modelPathEdit_->text().size());
          });

  connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

AppSettings SettingsDialog::settings() const {
  AppSettings s;
  s.modelPath = modelPathEdit_->text().toStdString();
  s.nthreads = nthreadsSpin_->value();
  s.deviceIndex = deviceIndexSpin_->value();
  s.flip = flipCheck_->isChecked();
  s.selectROI = selectROICheck_->isChecked();
  s.maskMode = maskModeCombo_->currentData().toString().toStdString();
  s.colorModeType =
      colorModeTypeCombo_->currentData().toString().toStdString();
  s.colorHueTol = colorHueTolSpin_->value();
  s.colorSatTol = colorSatTolSpin_->value();
  s.colorValTol = colorValTolSpin_->value();
  s.modelSize = modelSizeSpin_->value();
  s.strength = static_cast<float>(strengthSpin_->value());
  s.softening = static_cast<float>(softeningSpin_->value());
  s.padFactor = padFactorSpin_->value();
  s.distortInside = distortInsideCheck_->isChecked();
  s.temporalSmooth = static_cast<float>(temporalSmoothSpin_->value());
  s.lowerRes = static_cast<float>(lowerResSpin_->value());
  s.backgroundsDir = backgroundsDirEdit_->text().toStdString();
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

void SettingsDialog::browseModelPath() {
  const QString path = QFileDialog::getOpenFileName(
      this, "Select Segmentation Model",
      modelPathEdit_->text(), "TorchScript models (*.pt *.pth);;All files (*)");
  if (!path.isEmpty()) {
    modelPathEdit_->setText(path);
    modelPathEdit_->setCursorPosition(modelPathEdit_->text().size());
  }
}
