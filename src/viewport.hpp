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

// Standard includes

// Qt includes
#include <qtwidgets/QLabel>
#include <qtwidgets/QMainWindow>

// External includes
#include <opencv2/opencv.hpp>

// Local includes
#include "backgrounds.hpp"
#include "cam_feed.hpp"

/**
 * @brief ViewPort class
 *
 * This class defines the UI for the GravyLensing application.
 *
 * It displays the input frame, mask, overlay, and lensed background in a grid
 * layout. It also provides methods to switch between different views and set
 * the background images.
 */
class ViewPort : public QMainWindow {
  Q_OBJECT
public:
  explicit ViewPort(QWidget *parent = nullptr);
  ~ViewPort() override;

  /// Display only the lensed background
  void showLensedView();

  /// Display 2×2 grid: input frame, mask, overlay, lensed background
  void showGridView();

  // Set the background images
  void setBackgroundImages(Backgrounds *backgrounds) {
    backgrounds_ = backgrounds;
  }

protected:
  // catch key presses
  void keyPressEvent(QKeyEvent *event) override;

public Q_SLOTS:
  void setImage(const cv::Mat &image);
  void setBackground(const cv::Mat &background);
  void setLens(const cv::Mat &lens);
  void setMask(const cv::Mat &mask);

private:
  QLabel *imageLabel_;
  QLabel *backgroundLabel_;
  QLabel *lensLabel_;
  QLabel *maskLabel_;

  cv::Mat image_;
  cv::Mat background_;
  cv::Mat lens_;
  cv::Mat mask_;

  // Pointer to all the available background images
  Backgrounds *backgrounds_{nullptr};
};

// The getter called from main.cpp
ViewPort *initViewport(Backgrounds *backgrounds, bool debugGrid);
