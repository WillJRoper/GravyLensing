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

// Qt includes
#include <QApplication>
#include <QGridLayout>
#include <QKeyEvent>
#include <QPixmap>
#include <QScreen>
#include <QSizePolicy>
#include <QVBoxLayout>

/**
 * @brief Convert cv::Mat to QImage without copying pixel data.
 *
 * This function converts a cv::Mat object to a QImage object
 * without copying the pixel data. It handles different color formats
 * such as CV_8UC3, CV_8UC1, and CV_8UC4.
 *
 * @param mat The cv::Mat object to convert.
 *
 * @return A QImage object representing the cv::Mat data.
 */
static QImage MatToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC3:
    // BGR888 exists in Qt6; no need to rgbSwap or copy
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

/**
 * @brief ViewPort constructor.
 *
 * This constructor initializes the ViewPort class and sets up
 * the labels for displaying images.
 *
 * @param parent The parent widget (default is nullptr).
 */
ViewPort::ViewPort(QWidget *parent)
    : QMainWindow(parent), imageLabel_(new QLabel),
      backgroundLabel_(new QLabel), lensLabel_(new QLabel),
      maskLabel_(new QLabel) {
  // Make labels scale their pixmaps to fit
  for (auto lbl : {imageLabel_, backgroundLabel_, lensLabel_, maskLabel_}) {
    lbl->setScaledContents(true);
    lbl->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  }

  // Make the main window the size of the screen
  showMaximized();
}

/**
 * @brief ViewPort destructor.
 *
 * This destructor cleans up the ViewPort class.
 */
ViewPort::~ViewPort() = default;

/**
 * @brief Set the image to be displayed in the viewport.
 *
 * This function sets the image to be displayed in the viewport
 * by converting it to a QImage and setting it as a pixmap.
 *
 * @param image The cv::Mat object representing the image.
 */
void ViewPort::setImage(const cv::Mat &image) {
  // 1) shallow‐copy header only (no pixel data copy)
  image_ = image;

  // 2) wrap image_.data in a QImage—no deep copy
  QImage qi = MatToQImage(image_);

  // 3) QPixmap::fromImage does the one necessary copy into the widget
  imageLabel_->setPixmap(QPixmap::fromImage(qi));
}

/**
 * @brief Set the background image to be displayed in the viewport.
 *
 * This function sets the background image to be displayed in the viewport
 * by converting it to a QImage and setting it as a pixmap.
 *
 * @param background The cv::Mat object representing the background image.
 */
void ViewPort::setBackground(const cv::Mat &background) {
  // shallow‐copy header only
  background_ = background;

  // wrap without copying
  QImage qi = MatToQImage(background_);

  // single copy into the widget's pixmap
  backgroundLabel_->setPixmap(QPixmap::fromImage(qi));
}

/**
 * @brief Set the lens image to be displayed in the viewport.
 *
 * This function sets the lens image to be displayed in the viewport
 * by converting it to a QImage and setting it as a pixmap.
 *
 * @param lens The cv::Mat object representing the lens image.
 */
void ViewPort::setLens(const cv::Mat &lens) {
  // shallow‐copy header only
  lens_ = lens;

  // wrap without copying
  QImage qi = MatToQImage(lens_);

  // single copy into the widget's pixmap
  lensLabel_->setPixmap(QPixmap::fromImage(qi));
}

void ViewPort::setMask(const cv::Mat &mask) {
  // shallow‐copy header only
  mask_ = mask;

  QImage qi = MatToQImage(mask_);
  maskLabel_->setPixmap(QPixmap::fromImage(qi));
}

/**
 * @brief Show the lensed view in the viewport.
 *
 * This function sets up the viewport to display only the lensed image.
 */
void ViewPort::showLensedView() {
  // Single‐pane view
  QWidget *w = new QWidget(this);
  auto *layout = new QVBoxLayout(w);
  layout->addWidget(lensLabel_);
  setCentralWidget(w);

  // Resize window to exactly the lensed‐image resolution
  if (!lens_.empty()) {
    // note: cols = width, rows = height
    resize(lens_.cols, lens_.rows);
  }
}

/**
 * @brief Show the grid view in the viewport.
 *
 * This function sets up the viewport to display a 2×2 grid of images:
 * raw frame, mask, overlay, and lensed background.
 */
void ViewPort::showGridView() {
  // Create a new central widget and 2×2 grid layout
  QWidget *w = new QWidget(this);
  auto *grid = new QGridLayout(w);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setSpacing(0);

  // Add only the four image labels:
  grid->addWidget(imageLabel_, 0, 0);      // top-left: raw frame
  grid->addWidget(maskLabel_, 0, 1);       // top-right: mask
  grid->addWidget(backgroundLabel_, 1, 0); // bottom-left: overlay
  grid->addWidget(lensLabel_, 1, 1);       // bottom-right: lensed

  setCentralWidget(w);

  // Resize window to exactly fit 2×2 of the frame size
  if (!image_.empty()) {
    int w0 = image_.cols;
    int h0 = image_.rows;
    resize(w0 * 2, h0 * 2);
  }
}

/**
 * @brief Handle key press events in the viewport.
 *
 * This function handles key press events in the viewport.
 * It allows switching between background images and quitting the application.
 *
 * @param event The QKeyEvent object representing the key press event.
 */
void ViewPort::keyPressEvent(QKeyEvent *event) {
  int k = event->key();

  // Exit condition
  if (k == Qt::Key_Escape) {
    qApp->quit();
  }

  // Background swapping
  else if (k >= Qt::Key_0 && k <= Qt::Key_9) {
    // Map '0'..'9' → 0..9
    size_t idx = static_cast<size_t>(k - Qt::Key_0);

    // If we don't have a background object, bail!
    if (!backgrounds_) {
      std::cerr << "No background attached to viewport." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Cool, we have what we expect, switch the background (which will have
    // a knock on effect of updating everything else via signals). Throw an
    // error if it failed.
    if (!backgrounds_->setIndex(idx)) {
      std::cerr << "No background loaded at index " << idx << "; only have "
                << backgrounds_->size() << " images." << std::endl;
      std::cerr << "Press '0'..'" << backgrounds_->size() - 1
                << "' to switch backgrounds." << std::endl;
    }
  }

  // Let Qt handle anything else (arrows, function keys, etc.)
  else {
    QMainWindow::keyPressEvent(event);
  }
}

/**
 * @brief Setup the viewport with the specified background images.
 *
 * This function sets up the viewport with the specified background images
 * and displays the first image.
 *
 * @param backgrounds The Backgrounds object containing the background images.
 *
 * @return The VeiwPort object.
 */
ViewPort *initViewport(Backgrounds *backgrounds, bool debugGrid) {

  // Create the viewport
  ViewPort *vp = new ViewPort();

  // Set the view port title
  vp->setWindowTitle("GravyLensing");

  // Attach the backgrounds instance
  vp->setBackgroundImages(backgrounds);

  // Show the appropriate view (when debugging we show a gird of 4 images
  // to help debug what the program is doing)
  if (debugGrid) {
    vp->showGridView();
  } else {
    vp->showLensedView();
  }

  // Show it, we're up and running!
  vp->show();

  return vp;
}
