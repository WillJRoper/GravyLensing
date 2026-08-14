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
#include <QPixmap>
#include <QScreen>
#include <QSizePolicy>
#include <QVBoxLayout>

#include "perf_log.hpp"
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

  QAction *selectROIAction = fileMenu->addAction("Select &Region...");
  selectROIAction->setShortcut(QKeySequence("Shift+R"));
  selectROIAction->setToolTip("Draw a region of interest on the camera feed.");
  connect(selectROIAction, &QAction::triggered, this,
          &ViewPort::selectROIRequested);

  selectColorAction_ = fileMenu->addAction("Select &Color...");
  selectColorAction_->setShortcut(QKeySequence("Shift+S"));
  selectColorAction_->setToolTip("Pick an HSV colour to track.");
  connect(selectColorAction_, &QAction::triggered, this,
          &ViewPort::selectColorRequested);

  fileMenu->addSeparator();

  QAction *quitAction = fileMenu->addAction("&Quit");
  quitAction->setShortcut(QKeySequence::Quit);
  quitAction->setMenuRole(QAction::QuitRole);
  connect(quitAction, &QAction::triggered, qApp, &QApplication::quit);

  // ── View ──────────────────────────────────────────────────────────
  QMenu *viewMenu = mb->addMenu("&View");

  debugGridAction_ = viewMenu->addAction("Debug &Grid");
  debugGridAction_->setCheckable(true);
  debugGridAction_->setShortcut(QKeySequence("Shift+D"));
  debugGridAction_->setToolTip("Toggle the 2x2 diagnostic view.");
  connect(debugGridAction_, &QAction::toggled, this,
          &ViewPort::debugGridToggled);

  toggleMaskAction_ = viewMenu->addAction("Mask Mode: &Person");
  toggleMaskAction_->setShortcut(QKeySequence("Shift+M"));
  toggleMaskAction_->setToolTip("Switch AI person segmentation / HSV colour tracking.");
  connect(toggleMaskAction_, &QAction::triggered, this,
          &ViewPort::toggleMaskModeRequested);

  viewMenu->addSeparator();

  QMenu *bgMenu = viewMenu->addMenu("&Background");
  bgMenu->setToolTip("Quickly switch between loaded background images.");

  QAction *previousBackground = bgMenu->addAction("&Previous Background");
  previousBackground->setShortcut(Qt::Key_Left);
  connect(previousBackground, &QAction::triggered, this, [this]() {
    if (backgrounds_)
      backgrounds_->previous();
  });

  QAction *nextBackground = bgMenu->addAction("&Next Background");
  nextBackground->setShortcut(Qt::Key_Right);
  connect(nextBackground, &QAction::triggered, this, [this]() {
    if (backgrounds_)
      backgrounds_->next();
  });

  // ── Session ───────────────────────────────────────────────────────
  QMenu *settingsMenu = mb->addMenu("&Session");

  QAction *prefsAction = settingsMenu->addAction("Session &Settings...");
  prefsAction->setShortcut(QKeySequence::Preferences);
  prefsAction->setMenuRole(QAction::PreferencesRole);
  connect(prefsAction, &QAction::triggered, this, [this]() {
    SettingsDialog dlg(settings_, "Session Settings", "Restart Session",
                       targetHue_, targetSat_, targetVal_, hasTarget_,
                       hasROI_, roiX_, roiY_, roiW_, roiH_,
                       this);
    if (dlg.exec() == QDialog::Accepted) {
      if (dlg.roiSelectRequested()) {
        emit selectROIRequested();
        return;
      }
      if (dlg.roiClearRequested()) {
        hasROI_ = false;
        roiX_ = roiY_ = roiW_ = roiH_ = 0;
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
    }
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
  if (toggleMaskAction_)
    toggleMaskAction_->setText(isColorMode ? "Mask Mode: &Color"
                                           : "Mask Mode: &Person");
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
  lensLabel_->setPixmap(QPixmap::fromImage(MatToQImage(lens_)));
  const auto t1 = std::chrono::steady_clock::now();
  perf.addSample(std::chrono::duration<double, std::milli>(t1 - t0).count());
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
