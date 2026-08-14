#include "session_setup_dialog.hpp"

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFont>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>

#ifdef __APPLE__
#include "avfoundation_camera.hpp"
#endif
#include "backgrounds.hpp"
#include "settings_dialog.hpp"

namespace {

QLabel *description(const QString &text) {
  auto *label = new QLabel(text);
  label->setWordWrap(true);
  label->setForegroundRole(QPalette::PlaceholderText);
  label->setContentsMargins(24, 0, 0, 4);
  return label;
}

} // namespace

SessionSetupDialog::SessionSetupDialog(const AppSettings &settings,
                                       float currentHue, float currentSat,
                                       float currentVal, bool hasTarget,
                                       bool hasROI, QWidget *parent)
    : QDialog(parent), settings_(settings), pickedHue_(currentHue),
      pickedSat_(currentSat), pickedVal_(currentVal), hasTarget_(hasTarget),
      hasROI_(hasROI) {
  setWindowTitle("Start GravyLensing");
  setMinimumWidth(620);
  resize(680, 700);

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(24, 22, 24, 18);
  layout->setSpacing(16);

  auto *heading = new QLabel("Start a lensing session");
  QFont headingFont = heading->font();
  headingFont.setPointSize(22);
  headingFont.setWeight(QFont::DemiBold);
  heading->setFont(headingFont);
  layout->addWidget(heading);

  auto *intro = new QLabel(
      "Choose the subject and camera. Recommended defaults work well on most "
      "Apple Silicon Macs.");
  intro->setWordWrap(true);
  intro->setForegroundRole(QPalette::PlaceholderText);
  layout->addWidget(intro);

  auto *subjectGroup = new QGroupBox("What should create the lens?");
  auto *subjectLayout = new QVBoxLayout(subjectGroup);
  personRadio_ = new QRadioButton("People (Recommended)");
  personRadio_->setToolTip(
      "Automatically detect people. No manual target selection is required.");
  subjectLayout->addWidget(personRadio_);
  subjectLayout->addWidget(description(
      "Automatically finds people in the camera image. Best starting point."));
  colorRadio_ = new QRadioButton("A selected colour");
  colorRadio_->setToolTip(
      "Lens a colour selected from the live camera after starting.");
  subjectLayout->addWidget(colorRadio_);
  subjectLayout->addWidget(description(
      "Best for objects, clothing, props, or green-screen-style effects."));
  layout->addWidget(subjectGroup);

  auto *cameraGroup = new QGroupBox("Camera");
  auto *cameraForm = new QFormLayout(cameraGroup);
  cameraForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
#ifdef __APPLE__
  cameraCombo_ = new QComboBox;
  cameraCombo_->setToolTip("Camera used for the live feed.");
  cameraForm->addRow("Camera", cameraCombo_);
  refreshCameras();
  auto *cameraTimer = new QTimer(this);
  connect(cameraTimer, &QTimer::timeout, this,
          &SessionSetupDialog::refreshCameras);
  cameraTimer->start(1000);
#else
  cameraSpin_ = new QSpinBox;
  cameraSpin_->setRange(0, 99);
  cameraSpin_->setToolTip("Camera 0 is normally the built-in camera.");
  cameraForm->addRow("Camera number", cameraSpin_);
#endif

  fpsCombo_ = new QComboBox;
  fpsCombo_->setToolTip(
      "15 fps uses less power, 30 fps is recommended, and 60 fps is smoothest "
      "when supported by the camera.");
  fpsCombo_->addItem("15 fps - Lower power", 15);
  fpsCombo_->addItem("30 fps - Recommended", 30);
  fpsCombo_->addItem("60 fps - Smoothest", 60);
  fpsCombo_->addItem("Custom...", -1);
  customFpsSpin_ = new QSpinBox;
  customFpsSpin_->setRange(1, 240);
  customFpsSpin_->setSuffix(" fps");
  customFpsSpin_->setToolTip("Request a specific frame rate from the camera.");
  auto *fpsRow = new QHBoxLayout;
  fpsRow->addWidget(fpsCombo_, 1);
  fpsRow->addWidget(customFpsSpin_);
  cameraForm->addRow("Frame rate", fpsRow);
  connect(fpsCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
    customFpsSpin_->setVisible(fpsCombo_->currentData().toInt() < 0);
  });

  mirrorCheck_ = new QCheckBox("Mirror the camera image");
  mirrorCheck_->setToolTip(
      "Make the preview behave like a mirror so movements feel natural.");
  cameraForm->addRow("Preview", mirrorCheck_);
  layout->addWidget(cameraGroup);

  auto *sessionGroup = new QGroupBox("Session");
  auto *sessionForm = new QFormLayout(sessionGroup);
  sessionForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
  qualityCombo_ = new QComboBox;
  qualityCombo_->setToolTip(
      "Balanced is recommended. Fast improves frame rate; High improves mask "
      "detail at greater processing cost.");
  qualityCombo_->addItem("Fast", "fast");
  qualityCombo_->addItem("Balanced (Recommended)", "balanced");
  qualityCombo_->addItem("High Quality", "high");
  sessionForm->addRow("Quality", qualityCombo_);

  selectRegionCheck_ = new QCheckBox("Limit the camera area");
  selectRegionCheck_->setToolTip(
      "After you press Start, choose the part of the camera image to use. Leave "
      "off to use the full frame.");
  sessionForm->addRow("Region", selectRegionCheck_);

  includedBackgroundsRadio_ = new QRadioButton("Included backgrounds");
  includedBackgroundsRadio_->setToolTip(
      "Use the background collection packaged with GravyLensing.");
  customBackgroundsRadio_ = new QRadioButton("Custom folder");
  customBackgroundsRadio_->setToolTip(
      "Use supported images from a folder on this Mac.");
  auto *backgroundSource = new QVBoxLayout;
  backgroundSource->addWidget(includedBackgroundsRadio_);
  backgroundSource->addWidget(customBackgroundsRadio_);
  sessionForm->addRow("Background source", backgroundSource);

  backgroundsEdit_ = new QLineEdit;
  backgroundsEdit_->setToolTip(
      "Folder scanned for background images. The included backgrounds are used "
      "by default.");
  auto *backgroundRow = new QHBoxLayout;
  backgroundRow->addWidget(backgroundsEdit_, 1);
  auto *browseBackgrounds = new QPushButton("Browse...");
  browseBackgrounds->setToolTip("Choose a folder containing background images.");
  backgroundRow->addWidget(browseBackgrounds);
  connect(browseBackgrounds, &QPushButton::clicked, this, [this]() {
    const QString dir = QFileDialog::getExistingDirectory(
        this, "Select Backgrounds Folder", backgroundsEdit_->text());
    if (!dir.isEmpty()) {
      customBackgroundsRadio_->setChecked(true);
      backgroundsEdit_->setText(dir);
    }
  });
  sessionForm->addRow("Custom folder", backgroundRow);
  backgroundStatus_ = new QLabel;
  backgroundStatus_->setWordWrap(true);
  backgroundStatus_->setForegroundRole(QPalette::PlaceholderText);
  sessionForm->addRow({}, backgroundStatus_);
  const auto syncBackgroundSource = [=]() {
    const bool custom = customBackgroundsRadio_->isChecked();
    sessionForm->setRowVisible(backgroundRow, custom);
    updateBackgroundStatus();
  };
  connect(customBackgroundsRadio_, &QRadioButton::toggled, this,
          [syncBackgroundSource](bool) { syncBackgroundSource(); });
  connect(backgroundsEdit_, &QLineEdit::textChanged, this,
          [this](const QString &) { updateBackgroundStatus(); });
  layout->addWidget(sessionGroup);

  auto *actions = new QHBoxLayout;
  auto *advanced = new QPushButton("Advanced Settings...");
  advanced->setToolTip("Open all effect, detection, and performance settings.");
  actions->addWidget(advanced);
  actions->addStretch(1);
  auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok |
                                       QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Ok)->setText("Start Session");
  buttons->button(QDialogButtonBox::Ok)->setDefault(true);
  actions->addWidget(buttons);
  layout->addLayout(actions);

  connect(advanced, &QPushButton::clicked, this, [this]() {
    updateSettingsFromControls();
    SettingsDialog dialog(settings_, "Advanced Settings", "Save",
                          pickedHue_, pickedSat_, pickedVal_, hasTarget_,
                          hasROI_, 0, 0, 0, 0, this);
    if (dialog.exec() != QDialog::Accepted)
      return;
    settings_ = dialog.settings();
    if (dialog.roiSelectRequested())
      settings_.selectROI = true;
    if (dialog.roiClearRequested()) {
      settings_.selectROI = false;
      hasROI_ = false;
    }
    if (dialog.colorPickRequested()) {
      pickedHue_ = dialog.pickedHue();
      pickedSat_ = dialog.pickedSat();
      pickedVal_ = dialog.pickedVal();
      hasTarget_ = true;
      colorPickRequested_ = true;
    }
    if (dialog.colorFramePickRequested()) {
      settings_.maskMode = "color";
      hasTarget_ = false;
      colorFramePickRequested_ = true;
    }
    applySettingsToControls();
  });
  connect(buttons, &QDialogButtonBox::accepted, this, [this]() {
    updateSettingsFromControls();
    accept();
  });
  connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

  applySettingsToControls();
  syncBackgroundSource();
}

void SessionSetupDialog::applySettingsToControls() {
  colorRadio_->setChecked(settings_.maskMode == "color");
  personRadio_->setChecked(settings_.maskMode != "color");
#ifdef __APPLE__
  const int camera = cameraCombo_->findData(settings_.deviceIndex);
  if (camera >= 0)
    cameraCombo_->setCurrentIndex(camera);
#else
  cameraSpin_->setValue(settings_.deviceIndex);
#endif
  const int fps = fpsCombo_->findData(settings_.fps);
  fpsCombo_->setCurrentIndex(fps >= 0 ? fps : fpsCombo_->count() - 1);
  customFpsSpin_->setValue(settings_.fps);
  customFpsSpin_->setVisible(fps < 0);
  mirrorCheck_->setChecked(settings_.flip);
  selectRegionCheck_->setChecked(settings_.selectROI);
  const int quality =
      qualityCombo_->findData(QString::fromStdString(settings_.qualityMode));
  qualityCombo_->setCurrentIndex(quality >= 0 ? quality : 1);
  backgroundsEdit_->setText(QString::fromStdString(settings_.backgroundsDir));
  const bool included =
      settings_.backgroundsDir == AppSettings().backgroundsDir;
  includedBackgroundsRadio_->setChecked(included);
  customBackgroundsRadio_->setChecked(!included);
  updateBackgroundStatus();
}

void SessionSetupDialog::updateSettingsFromControls() {
  settings_.maskMode = colorRadio_->isChecked() ? "color" : "person";
#ifdef __APPLE__
  settings_.deviceIndex = cameraCombo_->currentData().toInt();
#else
  settings_.deviceIndex = cameraSpin_->value();
#endif
  const int guidedFps = fpsCombo_->currentData().toInt();
  settings_.fps = guidedFps > 0 ? guidedFps : customFpsSpin_->value();
  settings_.flip = mirrorCheck_->isChecked();
  settings_.selectROI = selectRegionCheck_->isChecked();
  settings_.qualityMode = qualityCombo_->currentData().toString().toStdString();
  settings_.backgroundsDir = includedBackgroundsRadio_->isChecked()
                                 ? AppSettings().backgroundsDir
                                 : backgroundsEdit_->text().toStdString();
}

void SessionSetupDialog::updateBackgroundStatus() {
  const std::string dir = includedBackgroundsRadio_->isChecked()
                              ? AppSettings().backgroundsDir
                              : backgroundsEdit_->text().toStdString();
  const size_t count = Backgrounds::discoverableImageCount(dir);
  backgroundStatus_->setText(
      count > 0 ? QString("%1 supported image%2 found")
                      .arg(count)
                      .arg(count == 1 ? "" : "s")
                : "No supported images found in this location");
}

#ifdef __APPLE__
void SessionSetupDialog::refreshCameras() {
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
