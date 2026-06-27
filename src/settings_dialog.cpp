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
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QSlider>
#include <QSizePolicy>
#include <QVBoxLayout>

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
  setMinimumWidth(760);
  resize(920, 720);
  setSizeGripEnabled(true);

  auto *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(20, 20, 20, 14);
  mainLayout->setSpacing(12);

  auto *scrollArea = new QScrollArea(this);
  scrollArea->setWidgetResizable(true);
  scrollArea->setFrameShape(QFrame::NoFrame);
  mainLayout->addWidget(scrollArea, 1);

  auto *content = new QWidget(scrollArea);
  scrollArea->setWidget(content);

  auto *contentLayout = new QVBoxLayout(content);
  contentLayout->setContentsMargins(0, 0, 0, 0);
  contentLayout->setSpacing(12);

  auto *summary = new QLabel(
      "Choose the mask source first, then tune the relevant settings below. "
      "Changes apply when you start or restart the session.");
  summary->setWordWrap(true);
  contentLayout->addWidget(summary);

  auto *columns = new QHBoxLayout;
  columns->setSpacing(24);
  contentLayout->addLayout(columns);

  auto *leftColumn = new QVBoxLayout;
  auto *rightColumn = new QVBoxLayout;
  columns->addLayout(leftColumn, 1);
  columns->addLayout(rightColumn, 1);

  // ── Camera group ────────────────────────────────────────────────────
  auto *cameraGroup = new QGroupBox("Camera");
  auto *cameraForm = new QFormLayout(cameraGroup);
  configureFormLayout(cameraForm);

  deviceIndexSpin_ = makeIntSpin(0, 99, 1, {},
      "Which camera device to open (0 is the built-in webcam).");
  deviceIndexSpin_->setValue(settings.deviceIndex);
  addFormRow(cameraForm, "Device index", deviceIndexSpin_->toolTip(),
             deviceIndexSpin_);

  flipCheck_ = makeCheck("Mirror camera feed horizontally",
      "Flip the image so movement in the real world and on-screen are "
      "directionally consistent.");
  flipCheck_->setChecked(settings.flip);
  addFormRow(cameraForm, "Flip", flipCheck_->toolTip(), flipCheck_);

  selectROICheck_ = makeCheck("Show region selector at startup",
      "On the next pipeline start, open an interactive ROI selection "
      "window.");
  selectROICheck_->setChecked(settings.selectROI);
  addFormRow(cameraForm, "Region of interest", selectROICheck_->toolTip(),
             selectROICheck_);

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
  configureFormLayout(modeForm);

  maskModeCombo_ = new QComboBox;
  maskModeCombo_->setToolTip(
      "Person uses an AI segmentation model; Color tracks a user-selected "
      "HSV colour.");
  maskModeCombo_->addItem("Person (AI segmentation)", "person");
  maskModeCombo_->addItem("Color tracking", "color");
  if (settings.maskMode == "color")
    maskModeCombo_->setCurrentIndex(1);
  addFormRow(modeForm, "Mask mode", maskModeCombo_->toolTip(),
             maskModeCombo_);
  leftColumn->addWidget(modeGroup);

  // ── Person Detection group ──────────────────────────────────────────
  auto *personGroup = new QGroupBox("Person Detection");
  auto *personForm = new QFormLayout(personGroup);
  configureFormLayout(personForm);

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
    addFormRow(personForm, "Model path",
               "Path to the TorchScript segmentation model to load.", row);
    connect(browseBtn_, &QPushButton::clicked, this,
            &SettingsDialog::browseModelPath);
    connect(modelPathEdit_, &QLineEdit::textChanged, this,
            [this](const QString &text) { modelPathEdit_->setToolTip(text); });
  }

  modelSizeSpin_ = makeIntSpin(128, 1024, 128, " px",
      "Larger models capture finer mask detail but run slower.");
  modelSizeSpin_->setValue(settings.modelSize);
  addFormRow(personForm, "Model size", modelSizeSpin_->toolTip(),
             modelSizeSpin_);

  temporalSmoothSpin_ = makeDoubleSpin(0.0, 1.0, 0.05, 2, {},
      "Higher values blend the mask more heavily with the previous frame, "
      "reducing flicker at the cost of responsiveness.");
  temporalSmoothSpin_->setValue(static_cast<double>(settings.temporalSmooth));
  addFormRow(personForm, "Temporal smooth", temporalSmoothSpin_->toolTip(),
             temporalSmoothSpin_);

  leftColumn->addWidget(personGroup);

  // ── Color Detection (subtype + tolerances) ─────────────────────────
  auto *colorGroup = new QGroupBox("Color Detection");
  auto *colorLayout = new QVBoxLayout(colorGroup);
  auto *colorForm = new QFormLayout;
  configureFormLayout(colorForm);

  colorModeTypeCombo_ = new QComboBox;
  colorModeTypeCombo_->setToolTip(
      "Fixed Color Key thresholds the target HSV range directly. "
      "Tracked Color Blob adds blob-tracking heuristics.");
  colorModeTypeCombo_->addItem("Fixed Color Key", "fixed_key");
  colorModeTypeCombo_->addItem("Tracked Color Blob", "tracked_blob");
  if (settings.colorModeType == "tracked_blob")
    colorModeTypeCombo_->setCurrentIndex(1);
  addFormRow(colorForm, "Mode", colorModeTypeCombo_->toolTip(),
             colorModeTypeCombo_);

  colorHueTolSpin_ = makeIntSpin(1, 90, 1, {},
      "Wider values include more hues around the target. "
      "Hue wraps at 0/180.");
  colorHueTolSpin_->setValue(settings.colorHueTol);
  addFormRow(colorForm, "Hue tolerance", colorHueTolSpin_->toolTip(),
             colorHueTolSpin_);

  colorSatTolSpin_ = makeIntSpin(1, 255, 5, {},
      "Wider values include more saturation variation.");
  colorSatTolSpin_->setValue(settings.colorSatTol);
  addFormRow(colorForm, "Saturation tolerance", colorSatTolSpin_->toolTip(),
             colorSatTolSpin_);

  colorValTolSpin_ = makeIntSpin(1, 255, 5, {},
      "Wider values include more brightness variation.");
  colorValTolSpin_->setValue(settings.colorValTol);
  addFormRow(colorForm, "Value tolerance", colorValTolSpin_->toolTip(),
             colorValTolSpin_);

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
  configureFormLayout(lensForm);

  strengthSpin_ = makeDoubleSpin(0.0, 10.0, 0.05, 2, {},
      "Strength multiplier applied to the deflection field. Larger "
      "values produce more dramatic lensing.");
  strengthSpin_->setValue(static_cast<double>(settings.strength));
  addFormRow(lensForm, "Strength", strengthSpin_->toolTip(), strengthSpin_);

  softeningSpin_ = makeDoubleSpin(0.0, 200.0, 1.0, 0, " px",
      "Softens the deflection kernel so nearby background pixels are "
      "affected more smoothly.");
  softeningSpin_->setValue(static_cast<double>(settings.softening));
  addFormRow(lensForm, "Softening radius", softeningSpin_->toolTip(),
             softeningSpin_);

  padFactorSpin_ = makeIntSpin(1, 10, 1, {},
      "FFT padding multiplier. Larger values reduce wrap-around "
      "artifacts but use more memory.");
  padFactorSpin_->setValue(settings.padFactor);
  addFormRow(lensForm, "FFT pad factor", padFactorSpin_->toolTip(),
             padFactorSpin_);

  lowerResSpin_ = makeDoubleSpin(0.1, 1.0, 0.1, 2, {},
      "Fraction of the background resolution at which lensing is "
      "computed. 1.0 = full resolution; 0.5 = half resolution (faster).");
  lowerResSpin_->setValue(static_cast<double>(settings.lowerRes));
  {
    auto *row = new QHBoxLayout;
    auto *lowerResSlider = new QSlider(Qt::Horizontal);
    lowerResSlider->setRange(10, 100);
    lowerResSlider->setSingleStep(5);
    lowerResSlider->setPageStep(10);
    lowerResSlider->setToolTip(lowerResSpin_->toolTip());
    lowerResSlider->setValue(
        static_cast<int>(std::round(lowerResSpin_->value() * 100.0)));
    row->addWidget(lowerResSlider, 1);
    row->addWidget(lowerResSpin_);

    connect(lowerResSlider, &QSlider::valueChanged, this,
            [this](int value) {
              const double scaled = static_cast<double>(value) / 100.0;
              if (!qFuzzyCompare(lowerResSpin_->value(), scaled))
                lowerResSpin_->setValue(scaled);
            });
    connect(lowerResSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, [lowerResSlider](double value) {
              const int scaled = static_cast<int>(std::round(value * 100.0));
              if (lowerResSlider->value() != scaled)
                lowerResSlider->setValue(scaled);
            });

    addFormRow(lensForm, "Resolution scale", lowerResSpin_->toolTip(), row);
  }

  distortInsideCheck_ = makeCheck("Distort inside the mask",
      "When enabled, the interior of the mask is also lensed (not only "
      "the background around it).");
  distortInsideCheck_->setChecked(settings.distortInside);
  addFormRow(lensForm, "Inside mask", distortInsideCheck_->toolTip(),
             distortInsideCheck_);

  rightColumn->addWidget(lensGroup);

  // ── Backgrounds group ──────────────────────────────────────────────
  auto *bgGroup = new QGroupBox("Backgrounds");
  auto *bgForm = new QFormLayout(bgGroup);
  configureFormLayout(bgForm);

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
    addFormRow(bgForm, "Directory",
               "Folder containing the backgrounds exposed in the UI.", row);
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
  addFormRow(bgForm, "Auto-cycle", autoCycleCheck_->toolTip(),
             autoCycleCheck_);

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
  addFormRow(bgForm, "Cycle interval", secondsPerBackgroundSpin_->toolTip(),
             secondsPerBackgroundSpin_);

  rightColumn->addWidget(bgGroup);

  // ── Performance group ───────────────────────────────────────────────
  auto *perfGroup = new QGroupBox("Performance");
  auto *perfForm = new QFormLayout(perfGroup);
  configureFormLayout(perfForm);

  nthreadsSpin_ = makeIntSpin(2, 256, 1, {},
      "Number of CPU worker threads (Qt reserves 3; the remainder are "
      "used for FFT and segmentation work).");
  nthreadsSpin_->setValue(settings.nthreads);
  addFormRow(perfForm, "CPU threads", nthreadsSpin_->toolTip(),
             nthreadsSpin_);

  debugGridCheck_ = makeCheck("Show diagnostic grid",
      "Display the 2×2 debug view (camera, mask, background, lensed) "
      "instead of the lensed-only view.");
  debugGridCheck_->setChecked(settings.debugGrid);
  addFormRow(perfForm, "Debug grid", debugGridCheck_->toolTip(),
             debugGridCheck_);

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

  const auto syncModeGroups = [=]() {
    const bool colorMode =
        maskModeCombo_->currentData().toString() == QLatin1String("color");
    personGroup->setEnabled(!colorMode);
    colorGroup->setEnabled(colorMode);
    personGroup->setFlat(colorMode);
    colorGroup->setFlat(!colorMode);
  };
  connect(maskModeCombo_, &QComboBox::currentIndexChanged, this,
          [=](int) { syncModeGroups(); });
  syncModeGroups();

  // ── Button box ──────────────────────────────────────────────────────
  auto *buttons = new QDialogButtonBox(
      QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok |
      QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Ok)->setText(acceptLabel);
  mainLayout->addWidget(buttons);

  connect(buttons->button(QDialogButtonBox::RestoreDefaults),
          &QPushButton::clicked, this, [this, syncModeGroups]() {
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

            // Reset ROI and colour-pick state
            hasROI_ = false;
            roiX_ = roiY_ = roiW_ = roiH_ = 0;
            roiInfoLabel_->setText("No region selected — full frame in use.");
            colorPickRequested_ = false;
             pickedHue_ = pickedSat_ = pickedVal_ = 0;
             updateSwatchDisplay(false);
             debugGridCheck_->setChecked(defaults.debugGrid);
             modelPathEdit_->setCursorPosition(
                 modelPathEdit_->text().size());
             syncModeGroups();
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
