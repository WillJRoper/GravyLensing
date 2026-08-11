/**
 * @file viewport.hpp
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

#pragma once

// Qt includes
#include <qtwidgets/QLabel>
#include <qtwidgets/QMainWindow>

#include <QAction>
#include <QMenu>

// External includes
#include <opencv2/opencv.hpp>

// Local includes
#include "backgrounds.hpp"
#include "cam_feed.hpp"
#include "settings.hpp"

class ViewPort : public QMainWindow {
  Q_OBJECT
public:
  explicit ViewPort(const AppSettings &settings, QWidget *parent = nullptr);
  ~ViewPort() override;

  void setBackgroundImages(Backgrounds *backgrounds);

  void setColorModeActive(bool active);

  void setSettings(const AppSettings &settings) { settings_ = settings; }

  void setDebugGridEnabled(bool enabled);

  AppSettings currentSettings() const { return settings_; }

  void setColorTarget(float h, float s, float v, bool has) {
    targetHue_ = h; targetSat_ = s; targetVal_ = v; hasTarget_ = has;
  }
  float targetHue() const { return targetHue_; }
  float targetSat() const { return targetSat_; }
  float targetVal() const { return targetVal_; }
  bool hasColorTarget() const { return hasTarget_; }

  void setROIState(bool has, int x, int y, int w, int h) {
    hasROI_ = has; roiX_ = x; roiY_ = y; roiW_ = w; roiH_ = h;
  }
  bool hasROI() const { return hasROI_; }
  int roiX() const { return roiX_; }
  int roiY() const { return roiY_; }
  int roiW() const { return roiW_; }
  int roiH() const { return roiH_; }

public Q_SLOTS:
  /// Sync the View > Debug Grid checkmark without re-emitting toggled().
  void setDebugGridChecked(bool checked);

  /// Update the View > Mask Mode label to reflect the current mode.
  void setMaskModeLabel(bool isColorMode);

  // Image data slots — called from worker threads via Qt::QueuedConnection.
  void setImage(const cv::Mat &image);
  void setBackground(const cv::Mat &background);
  void setLens(const cv::Mat &lens);
  void setMask(const cv::Mat &mask);

signals:
  void selectROIRequested();
  void selectColorRequested();
  void toggleMaskModeRequested();
  void debugGridToggled(bool enabled);
  void settingsChanged(const AppSettings &newSettings);

protected:
  void keyPressEvent(QKeyEvent *event) override;

private:
  void setupMenuBar();
  void setupViewLayout();

  QLabel *imageLabel_;
  QLabel *backgroundLabel_;
  QLabel *lensLabel_;
  QLabel *maskLabel_;

  cv::Mat image_;
  cv::Mat background_;
  cv::Mat lens_;
  cv::Mat mask_;

  Backgrounds *backgrounds_{nullptr};
  AppSettings settings_;
  float targetHue_ = 0;
  float targetSat_ = 0;
  float targetVal_ = 0;
  bool hasTarget_ = false;
  bool hasROI_ = false;
  int roiX_ = 0, roiY_ = 0, roiW_ = 0, roiH_ = 0;

  QAction *selectColorAction_ = nullptr;
  QAction *toggleMaskAction_ = nullptr;
  QAction *debugGridAction_ = nullptr;

};

ViewPort *initViewport(Backgrounds *backgrounds, const AppSettings &settings,
                       bool debugGrid);
