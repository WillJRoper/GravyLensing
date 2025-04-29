
#include "viewport.hpp"

#include <QGridLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QPixmap>
#include <QScreen>
#include <QSizePolicy>
#include <QVBoxLayout>

static QImage MatToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  case CV_8UC3: {
    // BGR → RGB
    QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_BGR888);
    return img.copy();
  }
  case CV_8UC1: {
    QImage img(mat.data, mat.cols, mat.rows, mat.step,
               QImage::Format_Grayscale8);
    return img.copy();
  }
  case CV_8UC4: {
    QImage img(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
    return img.copy();
  }
  default:
    return QImage();
  }
}

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

ViewPort::~ViewPort() = default;

void ViewPort::setImage(const cv::Mat &image) {
  image_ = image.clone();
  auto qi = MatToQImage(image_);
  imageLabel_->setPixmap(QPixmap::fromImage(qi));
}

void ViewPort::setBackground(const cv::Mat &background) {
  background_ = background.clone();
  auto qi = MatToQImage(background_);
  backgroundLabel_->setPixmap(QPixmap::fromImage(qi));
}

void ViewPort::setLens(const cv::Mat &lens) {
  lens_ = lens.clone();
  auto qi = MatToQImage(lens_);
  lensLabel_->setPixmap(QPixmap::fromImage(qi));
}

void ViewPort::setMask(const cv::Mat &mask) {
  // mask is binary CV_8UC1; colorize to RGB red mask if you like, or show as
  // gray
  mask_ = mask.clone();
  auto qi = MatToQImage(mask_);
  maskLabel_->setPixmap(QPixmap::fromImage(qi));
}

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

void ViewPort::keyPressEvent(QKeyEvent *event) {
  int k = event->key();

  // Exit condition
  if (k == Qt::Key_Escape) {
    close(); // this will make isVisible() return false

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

    // Cool, we have what we expect. Switch background (this will also
    // remap the lens and camera feed to the background dimensions)
    if (backgrounds_->setIndex(idx)) {
      {
        std::lock_guard lk(lensObj_->geometryMutex_);
        setBackground(backgrounds_->current());
        lensObj_->updateGeometry(backgrounds_->cols(), backgrounds_->rows());
      }
    }

    // Bad things have happened...
    else {
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
