/**
 * @file viewport.cpp
 *
 * This file defines the UI for the GravyLensing application.
 *
 * This class loads a set of background images from a specified directory
 * and enables switching between them.
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
#include "viewport.hpp"

#include <QApplication>
#include <QGridLayout>
#include <QKeyEvent>
#include <QMenuBar>
#include <QMessageBox>
#include <QPixmap>
#include <QScreen>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "perf_log.hpp"
#include "image_compositing.hpp"
#include "settings_dialog.hpp"

static QImage MatToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC3:
    return QImage(mat.data, mat.cols, mat.rows, int(mat.step),
                  QImage::Format_BGR888);
  case CV_8UC1:
    return QImage(mat.data, mat.cols, mat.rows, int(mat.step),
                  QImage::Format_Grayscale8);
  case CV_8UC4:
    return QImage(mat.data, mat.cols, mat.rows, int(mat.step),
                  QImage::Format_ARGB32);
  default:
    return QImage();
  }
}

ViewPort::ViewPort(const AppSettings &settings, QWidget *parent)
    : QMainWindow(parent), imageLabel_(new QLabel(this)),
      backgroundLabel_(new QLabel(this)), lensLabel_(new QLabel(this)),
      maskLabel_(new QLabel(this)), settings_(settings) {

  for (auto lbl : {imageLabel_, backgroundLabel_, lensLabel_, maskLabel_}) {
    lbl->setScaledContents(true);
    lbl->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  }

  setupViewLayout();
  setupMenuBar();
  showMaximized();
}

ViewPort::~ViewPort() = default;

// ─────────────────────────────────────────────────────────────────────────────
//  View layout  (single grid — visibility toggle, no widgets destroyed)
// ─────────────────────────────────────────────────────────────────────────────

void ViewPort::setupViewLayout() {
  QWidget *w = new QWidget(this);
  auto *grid = new QGridLayout(w);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setSpacing(0);

  imageLabel_->setVisible(false);
  maskLabel_->setVisible(false);
  backgroundLabel_->setVisible(false);

  grid->addWidget(imageLabel_, 0, 0);
  grid->addWidget(maskLabel_, 0, 1);
  grid->addWidget(backgroundLabel_, 1, 0);
  // lensLabel_ starts spanning all cells (full window)
  grid->addWidget(lensLabel_, 0, 0, 2, 2);

  setCentralWidget(w);
}

void ViewPort::setDebugGridEnabled(bool enabled) {
  QGridLayout *grid =
      qobject_cast<QGridLayout *>(centralWidget()->layout());
  if (!grid) return;

  if (enabled) {
    // Move lensLabel back to its single cell
    grid->addWidget(lensLabel_, 1, 1);
    imageLabel_->setVisible(true);
    maskLabel_->setVisible(true);
    backgroundLabel_->setVisible(true);
  } else {
    // Span lensLabel across the full 2×2 grid
    grid->addWidget(lensLabel_, 0, 0, 2, 2);
    imageLabel_->setVisible(false);
    maskLabel_->setVisible(false);
    backgroundLabel_->setVisible(false);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Menu bar
// ─────────────────────────────────────────────────────────────────────────────

void ViewPort::setupMenuBar() {
  QMenuBar *mb = menuBar();

  // ── File ──────────────────────────────────────────────────────────
  QMenu *fileMenu = mb->addMenu("&File");
  QAction *quitAction = fileMenu->addAction("&Quit");
  quitAction->setShortcut(QKeySequence::Quit);
  quitAction->setMenuRole(QAction::QuitRole);
  connect(quitAction, &QAction::triggered, qApp, &QApplication::quit);

  // ── Session ───────────────────────────────────────────────────────
  QMenu *sessionMenu = mb->addMenu("&Session");
  QAction *settingsAction = sessionMenu->addAction("&Settings...");
  settingsAction->setShortcut(QKeySequence::Preferences);
  settingsAction->setMenuRole(QAction::NoRole);
  settingsAction->setToolTip("Change session, camera, lens, and performance settings.");
  connect(settingsAction, &QAction::triggered, this, [this]() {
    SettingsDialog dlg(settings_, "Settings", "Restart Session", targetHue_,
                       targetSat_, targetVal_, hasTarget_, hasROI_, roiX_, roiY_,
                       roiW_, roiH_, this);
    if (dlg.exec() != QDialog::Accepted)
      return;
    if (dlg.roiSelectRequested()) {
      emit selectROIRequested();
      return;
    }
    if (dlg.roiClearRequested()) {
      setROIState(false, 0, 0, 0, 0);
      emit clearROIRequested();
    }
    if (dlg.colorFramePickRequested()) {
      settings_ = dlg.settings();
      emit settingsChanged(settings_);
      emit selectColorRequested();
      return;
    }
    if (dlg.colorPickRequested()) {
      targetHue_ = dlg.pickedHue();
      targetSat_ = dlg.pickedSat();
      targetVal_ = dlg.pickedVal();
      hasTarget_ = true;
    }
    settings_ = dlg.settings();
    emit settingsChanged(settings_);
  });

  // ── Lens ──────────────────────────────────────────────────────────
  QMenu *lensMenu = mb->addMenu("&Lens");
  auto *lensModeGroup = new QActionGroup(this);
  lensModeGroup->setExclusive(true);
  personLensAction_ = lensMenu->addAction("&People");
  personLensAction_->setCheckable(true);
  personLensAction_->setChecked(settings_.maskMode != "color");
  lensModeGroup->addAction(personLensAction_);
  colorLensAction_ = lensMenu->addAction("Selected &Colour");
  colorLensAction_->setCheckable(true);
  colorLensAction_->setChecked(settings_.maskMode == "color");
  lensModeGroup->addAction(colorLensAction_);
  connect(personLensAction_, &QAction::triggered, this,
          [this]() { emit maskModeRequested(false); });
  connect(colorLensAction_, &QAction::triggered, this,
          [this]() { emit maskModeRequested(true); });

  toggleMaskAction_ = lensMenu->addAction("&Switch Lens Mode");
  toggleMaskAction_->setShortcut(QKeySequence("Shift+M"));
  connect(toggleMaskAction_, &QAction::triggered, this,
          &ViewPort::toggleMaskModeRequested);
  lensMenu->addSeparator();

  selectColorAction_ = lensMenu->addAction("Select &Colour...");
  selectColorAction_->setShortcut(QKeySequence("Shift+S"));
  connect(selectColorAction_, &QAction::triggered, this,
          &ViewPort::selectColorRequested);

  QAction *selectROIAction = lensMenu->addAction("Select &Region...");
  selectROIAction->setShortcut(QKeySequence("Shift+R"));
  connect(selectROIAction, &QAction::triggered, this,
          &ViewPort::selectROIRequested);
  clearROIAction_ = lensMenu->addAction("Use &Full Frame");
  clearROIAction_->setEnabled(hasROI_);
  connect(clearROIAction_, &QAction::triggered, this, [this]() {
    setROIState(false, 0, 0, 0, 0);
    emit clearROIRequested();
  });

  // ── Background ────────────────────────────────────────────────────
  QMenu *backgroundMenu = mb->addMenu("&Background");
  QAction *previousBackground = backgroundMenu->addAction("&Previous");
  previousBackground->setShortcut(Qt::Key_Left);
  connect(previousBackground, &QAction::triggered, this, [this]() {
    if (backgrounds_)
      backgrounds_->previous();
  });
  QAction *nextBackground = backgroundMenu->addAction("&Next");
  nextBackground->setShortcut(Qt::Key_Right);
  connect(nextBackground, &QAction::triggered, this, [this]() {
    if (backgrounds_)
      backgrounds_->next();
  });
  backgroundMenu->addSeparator();

  autoCycleAction_ = backgroundMenu->addAction("&Automatic Cycling");
  autoCycleAction_->setCheckable(true);
  autoCycleAction_->setChecked(settings_.secondsPerBackground > 0);
  connect(autoCycleAction_, &QAction::toggled, this,
          &ViewPort::backgroundAutoCycleToggled);

  QMenu *intervalMenu = backgroundMenu->addMenu("Cycle &Every");
  cycleIntervalGroup_ = new QActionGroup(this);
  cycleIntervalGroup_->setExclusive(true);
  for (const int seconds : {5, 10, 15, 30, 60}) {
    QAction *interval = intervalMenu->addAction(
        seconds == 60 ? "1 minute" : QString("%1 seconds").arg(seconds));
    interval->setCheckable(true);
    interval->setData(seconds);
    interval->setChecked(settings_.secondsPerBackground == seconds);
    cycleIntervalGroup_->addAction(interval);
    connect(interval, &QAction::triggered, this,
            [this, seconds]() { emit backgroundIntervalRequested(seconds); });
  }

  // ── View ──────────────────────────────────────────────────────────
  QMenu *viewMenu = mb->addMenu("&View");
  debugGridAction_ = viewMenu->addAction("Debug &Grid");
  debugGridAction_->setCheckable(true);
  debugGridAction_->setShortcut(QKeySequence("Shift+D"));
  debugGridAction_->setToolTip("Toggle the 2x2 diagnostic view.");
  connect(debugGridAction_, &QAction::toggled, this,
          &ViewPort::debugGridToggled);

  showLensContentsAction_ = viewMenu->addAction("Show Camera Inside &Lens");
  showLensContentsAction_->setCheckable(true);
  showLensContentsAction_->setShortcut(QKeySequence("Shift+I"));
  showLensContentsAction_->setChecked(settings_.showLensContents);
  showLensContentsAction_->setToolTip(
      "Show the live camera inside the detected lens mask over the lensed "
      "background.");
  connect(showLensContentsAction_, &QAction::toggled, this,
          &ViewPort::showLensContentsToggled);

  QAction *fullScreenAction = viewMenu->addAction("Toggle &Full Screen");
  fullScreenAction->setShortcut(QKeySequence("Ctrl+Meta+F"));
  connect(fullScreenAction, &QAction::triggered, this, [this]() {
    isFullScreen() ? showNormal() : showFullScreen();
  });

  // ── Help ──────────────────────────────────────────────────────────
  QMenu *helpMenu = mb->addMenu("&Help");
  QAction *shortcutsAction = helpMenu->addAction("Keyboard &Shortcuts");
  connect(shortcutsAction, &QAction::triggered, this, [this]() {
    QMessageBox::information(
        this, "Keyboard Shortcuts",
        "Settings: Cmd+,\nSelect colour: Shift+S\nSelect region: Shift+R\n"
        "Switch lens mode: Shift+M\nShow camera inside lens: Shift+I\n"
        "Debug grid: Shift+D\n"
        "Previous / next background: Left / Right\nQuit: Cmd+Q or Esc");
  });
  QAction *aboutAction = helpMenu->addAction("&About Gravy Lensing");
  aboutAction->setMenuRole(QAction::AboutRole);
  connect(aboutAction, &QAction::triggered, this, [this]() {
    QMessageBox::about(this, "About Gravy Lensing",
                       "Gravy Lensing\n\nBe dark matter. Bend spacetime. Warp "
                       "reality.\n\nReal-time gravitational lensing powered "
                       "by Apple Vision, Metal, OpenCV, and FFTW.");
  });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Menu state slots
// ─────────────────────────────────────────────────────────────────────────────

void ViewPort::setDebugGridChecked(bool checked) {
  if (debugGridAction_) {
    debugGridAction_->blockSignals(true);
    debugGridAction_->setChecked(checked);
    debugGridAction_->blockSignals(false);
  }
}

void ViewPort::setMaskModeLabel(bool isColorMode) {
  if (personLensAction_)
    personLensAction_->setChecked(!isColorMode);
  if (colorLensAction_)
    colorLensAction_->setChecked(isColorMode);
  if (toggleMaskAction_)
    toggleMaskAction_->setText(isColorMode ? "Switch to &People"
                                           : "Switch to &Selected Colour");
}

void ViewPort::setColorModeActive(bool active) {
  if (selectColorAction_) {
    selectColorAction_->setEnabled(true);
    selectColorAction_->setText(active ? "Reselect &Color..."
                                      : "Select &Color...");
  }
}

void ViewPort::setBackgroundImages(Backgrounds *backgrounds) {
  backgrounds_ = backgrounds;
}

void ViewPort::setSettings(const AppSettings &settings) {
  settings_ = settings;
  setBackgroundCycleState(settings.secondsPerBackground);
  setShowLensContentsEnabled(settings.showLensContents);
}

void ViewPort::setROIState(bool has, int x, int y, int w, int h) {
  hasROI_ = has;
  roiX_ = x;
  roiY_ = y;
  roiW_ = w;
  roiH_ = h;
  if (clearROIAction_)
    clearROIAction_->setEnabled(has);
}

void ViewPort::setBackgroundCycleState(int seconds) {
  if (autoCycleAction_) {
    QSignalBlocker blocker(autoCycleAction_);
    autoCycleAction_->setChecked(seconds > 0);
  }
  if (!cycleIntervalGroup_)
    return;
  for (QAction *action : cycleIntervalGroup_->actions()) {
    QSignalBlocker blocker(action);
    action->setChecked(seconds > 0 && action->data().toInt() == seconds);
  }
}

void ViewPort::setShowLensContentsEnabled(bool enabled) {
  showLensContents_ = enabled;
  if (showLensContentsAction_) {
    QSignalBlocker blocker(showLensContentsAction_);
    showLensContentsAction_->setChecked(enabled);
  }
  updateLensDisplay();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Image display
// ─────────────────────────────────────────────────────────────────────────────

void ViewPort::setImage(const cv::Mat &image) {
  static PerfLog perf("ui-image", 120);
  const auto t0 = std::chrono::steady_clock::now();
  image_ = image;
  imageLabel_->setPixmap(QPixmap::fromImage(MatToQImage(image_)));
  const auto t1 = std::chrono::steady_clock::now();
  perf.addSample(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

void ViewPort::setBackground(const cv::Mat &background) {
  background_ = background;
  backgroundLabel_->setPixmap(QPixmap::fromImage(MatToQImage(background_)));
}

void ViewPort::setLens(const cv::Mat &lens) {
  static PerfLog perf("ui-lens", 120);
  const auto t0 = std::chrono::steady_clock::now();
  lens_ = lens;
  updateLensDisplay();
  const auto t1 = std::chrono::steady_clock::now();
  perf.addSample(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

void ViewPort::updateLensDisplay() {
  if (lens_.empty())
    return;
  if (!showLensContents_ || image_.empty() || mask_.empty()) {
    lensLabel_->setPixmap(QPixmap::fromImage(MatToQImage(lens_)));
    return;
  }

  const cv::Mat composite = compositeMaskedForeground(lens_, image_, mask_);
  lensLabel_->setPixmap(QPixmap::fromImage(MatToQImage(composite)));
}

void ViewPort::setMask(const cv::Mat &mask) {
  mask_ = mask;
  maskLabel_->setPixmap(QPixmap::fromImage(MatToQImage(mask_)));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Key press
// ─────────────────────────────────────────────────────────────────────────────

void ViewPort::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Escape) {
    qApp->quit();
    return;
  }
  QMainWindow::keyPressEvent(event);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Init helper
// ─────────────────────────────────────────────────────────────────────────────

ViewPort *initViewport(Backgrounds *backgrounds, const AppSettings &settings,
                       bool debugGrid) {
  ViewPort *vp = new ViewPort(settings);
  vp->setWindowTitle("GravyLensing");
  vp->setBackgroundImages(backgrounds);
  vp->setDebugGridEnabled(debugGrid);
  vp->setDebugGridChecked(debugGrid);
  return vp;
}
