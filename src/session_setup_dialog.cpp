#include "session_setup_dialog.hpp"

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFont>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPainter>
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

class SessionBanner final : public QWidget {
public:
  explicit SessionBanner(QWidget *parent = nullptr) : QWidget(parent) {
    setFixedHeight(152);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setAccessibleName("Gravy Lensing");
  }

protected:
  void paintEvent(QPaintEvent *) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    const QRectF bounds = rect().adjusted(0, 0, -1, -1);
    QLinearGradient background(bounds.topLeft(), bounds.bottomRight());
    background.setColorAt(0.0, QColor("#10182f"));
    background.setColorAt(0.55, QColor("#24204c"));
    background.setColorAt(1.0, QColor("#512b67"));
    painter.setPen(Qt::NoPen);
    painter.setBrush(background);
    painter.drawRoundedRect(bounds, 18, 18);

    const QPointF lens(width() - 112.0, height() / 2.0);
    painter.setBrush(Qt::NoBrush);
    for (int radius = 26; radius <= 82; radius += 18) {
      QColor ring("#d9a7ff");
      ring.setAlpha(105 - radius / 2);
      painter.setPen(QPen(ring, radius == 26 ? 3.0 : 1.5));
      painter.drawEllipse(lens, radius, radius);
    }
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(255, 255, 255, 230));
    painter.drawEllipse(lens, 8, 8);

    painter.setPen(Qt::white);
    QFont titleFont = font();
    titleFont.setPointSize(28);
    titleFont.setWeight(QFont::Bold);
    titleFont.setLetterSpacing(QFont::AbsoluteSpacing, 1.4);
    painter.setFont(titleFont);
    painter.drawText(QRectF(28, 30, width() - 180, 48),
                     Qt::AlignLeft | Qt::AlignVCenter, "GRAVY LENSING");

    QFont subtitleFont = font();
    subtitleFont.setPointSize(12);
    subtitleFont.setWeight(QFont::Medium);
    painter.setFont(subtitleFont);
    painter.setPen(QColor("#ddd7ef"));
    painter.drawText(QRectF(30, 82, width() - 190, 28),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     "Be dark matter. Bend spacetime. Warp reality.");
  }
};

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
  setWindowTitle("Gravy Lensing");
  setMinimumWidth(660);
  resize(720, 760);

  auto *layout = new QVBoxLayout(this);
  layout->setContentsMargins(24, 22, 24, 18);
  layout->setSpacing(16);

  layout->addWidget(new SessionBanner(this));

  auto *intro = new QLabel("Create a new session");
  QFont introFont = intro->font();
  introFont.setPointSize(16);
  introFont.setWeight(QFont::DemiBold);
  intro->setFont(introFont);
  intro->setWordWrap(true);
  layout->addWidget(intro);

  auto *introDescription = new QLabel(
      "Choose what creates the lens and how the camera should behave. "
      "Recommended settings work well on most Apple Silicon Macs.");
  introDescription->setWordWrap(true);
  introDescription->setForegroundRole(QPalette::PlaceholderText);
  layout->addWidget(introDescription);

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

  cameraResolutionCombo_ = new QComboBox;
  cameraResolutionCombo_->setToolTip(
      "Requested camera capture resolution. The closest format supported by "
      "the selected camera is used.");
  cameraResolutionCombo_->addItem("Automatic", QSize());
  cameraResolutionCombo_->addItem("480p - 854 x 480", QSize(854, 480));
  cameraResolutionCombo_->addItem("720p - 1280 x 720 (Recommended)",
                                  QSize(1280, 720));
  cameraResolutionCombo_->addItem("1080p - 1920 x 1080", QSize(1920, 1080));
  cameraForm->addRow("Capture resolution", cameraResolutionCombo_);

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
      const size_t count =
          Backgrounds::discoverableImageCount(dir.toStdString());
      const QString formats =
          "Images do not need to be TIFFs. Supported formats are PNG, JPEG, "
          "BMP, GIF, TIFF, WebP, and SVG when a decoder is available.";
      const QString resolution =
          QString("Images will be preprocessed and cached at %1 x %2. Higher "
                  "resolution and frame-rate targets require substantially "
                  "more processing power. Change this in Settings > "
                  "Backgrounds.")
              .arg(settings_.backgroundWidth)
              .arg(settings_.backgroundHeight);
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
  auto *advanced = new QPushButton("Settings...");
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
    SettingsDialog dialog(settings_, "Settings", "Save",
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
  const int cameraResolution = cameraResolutionCombo_->findData(
      QSize(settings_.cameraWidth, settings_.cameraHeight));
  cameraResolutionCombo_->setCurrentIndex(cameraResolution >= 0
                                               ? cameraResolution
                                               : 0);
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
  const QSize cameraSize = cameraResolutionCombo_->currentData().toSize();
  settings_.cameraWidth = cameraSize.width();
  settings_.cameraHeight = cameraSize.height();
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
  const QString sizeNote = QString("; cached at %1 x %2")
                               .arg(settings_.backgroundWidth)
                               .arg(settings_.backgroundHeight);
  backgroundStatus_->setText(count > 0
                                 ? QString("%1 supported image%2 found%3")
                                       .arg(count)
                                       .arg(count == 1 ? "" : "s")
                                       .arg(sizeNote)
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
