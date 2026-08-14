/**
 * @file backgrounds.cpp
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

// Standard includes
#include <filesystem>
#include <string>

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

// Local includes
#include "backgrounds.hpp"

namespace fs = std::filesystem;

namespace {
QString backgroundCacheDir() {
  return QStandardPaths::writableLocation(QStandardPaths::CacheLocation) +
         "/backgrounds";
}
} // namespace

// The supported image file extensions (lower-case, upper-case is handled
// by lower-case conversion)
const std::vector<std::string> Backgrounds::kImageExts = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".svg"};

/**
 * @brief Setup the Background images.
 *
 * This function initializes the background images from the specified directory.
 *
 * @param dir The directory containing the background images.
 *
 * @return The Backgrounds object containing the loaded images.
 */
Backgrounds *initBackgrounds(const std::string &dir, int width, int height,
                             const std::string &fitMode, bool forceRebuild) {
  Backgrounds *backgrounds =
      new Backgrounds(dir, width, height, fitMode, forceRebuild);
  if (!backgrounds->load()) {
    std::cerr << "Fatal: No images found in directory: " << dir << "\n";
    std::exit(EXIT_FAILURE);
  }
  return backgrounds;
}

/**
 * @brief Backgrounds constructor
 *
 * @param dir path to folder containing your images
 */
Backgrounds::Backgrounds(const std::string &dir, int width, int height,
                         const std::string &fitMode, bool forceRebuild)
    : dir_(dir), width_(width), height_(height), fitMode_(fitMode),
      forceRebuild_(forceRebuild) {}

size_t Backgrounds::discoverableImageCount(const std::string &dir) {
  size_t count = 0;
  try {
    if (!fs::exists(dir) || !fs::is_directory(dir))
      return 0;
    for (const auto &entry : fs::directory_iterator(dir)) {
      if (!entry.is_regular_file())
        continue;
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (std::find(kImageExts.begin(), kImageExts.end(), ext) !=
              kImageExts.end() &&
          cv::haveImageReader(entry.path().string()))
        ++count;
    }
  } catch (const fs::filesystem_error &) {
    return 0;
  }
  return count;
}

size_t Backgrounds::cacheImageCount() {
  return QDir(backgroundCacheDir()).entryList({"*.png"}, QDir::Files).size();
}

uint64_t Backgrounds::cacheSizeBytes() {
  uint64_t bytes = 0;
  const QDir dir(backgroundCacheDir());
  for (const QFileInfo &file : dir.entryInfoList({"*.png"}, QDir::Files))
    bytes += static_cast<uint64_t>(file.size());
  return bytes;
}

bool Backgrounds::clearCache() {
  QDir dir(backgroundCacheDir());
  return !dir.exists() || dir.removeRecursively();
}

/**
 * @brief Scan & load *all* images with “known” extensions.
 *
 * @return false if directory doesn’t exist or no images found.
 */
bool Backgrounds::load() {
  std::vector<std::string> paths;

  try {
    if (!fs::exists(dir_) || !fs::is_directory(dir_))
      return false;

    // collect all matching paths
    for (auto const &entry : fs::directory_iterator(dir_)) {
      if (!entry.is_regular_file())
        continue;

      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (std::find(kImageExts.begin(), kImageExts.end(), ext) !=
          kImageExts.end()) {
        paths.push_back(entry.path().string());
      }
    }
  } catch (const fs::filesystem_error &) {
    return false;
  }

  if (paths.empty())
    return false;

  // sort so numbering is stable
  std::sort(paths.begin(), paths.end());

  std::vector<std::string> validPaths;
  cv::Mat firstImage;
  for (auto const &p : paths) {
    cv::Mat img;
    std::string cachedPath;
    if (prepareImage(p, cachedPath, img)) {
      validPaths.push_back(cachedPath);
      if (firstImage.empty())
        firstImage = std::move(img);
    }
  }

  if (validPaths.empty())
    return false;

  paths_ = std::move(validPaths);
  currentImage_ = std::move(firstImage);
  currentIdx_ = 0;
  return true;
}

bool Backgrounds::setDirectory(const std::string &dir) {
  return setSource(dir, width_, height_, fitMode_);
}

bool Backgrounds::setSource(const std::string &dir, int width, int height,
                            const std::string &fitMode, bool forceRebuild) {
  const std::string previousDir = dir_;
  const int previousWidth = width_;
  const int previousHeight = height_;
  const std::string previousFitMode = fitMode_;
  dir_ = dir;
  width_ = width;
  height_ = height;
  fitMode_ = fitMode;
  forceRebuild_ = forceRebuild;
  if (!load()) {
    dir_ = previousDir;
    width_ = previousWidth;
    height_ = previousHeight;
    fitMode_ = previousFitMode;
    forceRebuild_ = false;
    return false;
  }
  forceRebuild_ = false;
  emit backgroundChanged(current());
  return true;
}

/**
 * @brief Load a single image by path
 *
 * @param path path to the image
 * @param out output cv::Mat
 *
 * @return true if the image was loaded successfully, false otherwise
 */
bool Backgrounds::loadImage(const std::string &path, cv::Mat &out) const {
  out = cv::imread(path, cv::IMREAD_COLOR);
  return !out.empty();
}

bool Backgrounds::prepareImage(const std::string &path,
                               std::string &cachedPath, cv::Mat &out) const {
  if (width_ < 1 || height_ < 1)
    return false;

  const QFileInfo source(QString::fromStdString(path));
  static constexpr int kCacheSchemaVersion = 2;
  const QByteArray cacheKey =
      (source.absoluteFilePath() + "|" + QString::number(source.size()) + "|" +
       QString::number(source.lastModified().toMSecsSinceEpoch()) + "|" +
       QString::number(width_) + "x" + QString::number(height_) + "|" +
       QString::fromStdString(fitMode_) + "|v" +
       QString::number(kCacheSchemaVersion))
          .toUtf8();
  const QString cacheDir = backgroundCacheDir();
  if (!QDir().mkpath(cacheDir))
    return false;
  const QString cacheFile =
      cacheDir + "/" +
      QCryptographicHash::hash(cacheKey, QCryptographicHash::Sha256).toHex() +
      ".png";
  cachedPath = cacheFile.toStdString();

  if (!forceRebuild_ && loadImage(cachedPath, out) && out.cols == width_ &&
      out.rows == height_)
    return true;

  const cv::Mat sourceImage = cv::imread(path, cv::IMREAD_COLOR);
  if (sourceImage.empty())
    return false;

  if (fitMode_ == "stretch") {
    cv::resize(sourceImage, out, cv::Size(width_, height_), 0, 0,
               cv::INTER_AREA);
  } else {
    const double widthScale = static_cast<double>(width_) / sourceImage.cols;
    const double heightScale = static_cast<double>(height_) / sourceImage.rows;
    const double scale = fitMode_ == "fit" ? std::min(widthScale, heightScale)
                                            : std::max(widthScale, heightScale);
    cv::Mat resized;
    cv::resize(sourceImage, resized, {}, scale, scale,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);
    if (fitMode_ == "fit") {
      out = cv::Mat::zeros(height_, width_, CV_8UC3);
      const int x = (width_ - resized.cols) / 2;
      const int y = (height_ - resized.rows) / 2;
      resized.copyTo(out(cv::Rect(x, y, resized.cols, resized.rows)));
    } else {
      const int x = (resized.cols - width_) / 2;
      const int y = (resized.rows - height_) / 2;
      out = resized(cv::Rect(x, y, width_, height_)).clone();
    }
  }

  return cv::imwrite(cachedPath, out);
}

/**
 * @brief Get the currently-selected image.
 *
 * @return reference to the current image
 */
const cv::Mat &Backgrounds::current() const { return currentImage_; }

/**
 * @brief Advance to the next image (wraps round); returns false if none loaded.
 *
 * @return true if the next image was set successfully, false otherwise
 */
bool Backgrounds::next() {
  if (paths_.empty())
    return false;
  const size_t nextIdx = (currentIdx_ + 1) % paths_.size();
  cv::Mat nextImage;
  if (!loadImage(paths_[nextIdx], nextImage))
    return false;
  currentIdx_ = nextIdx;
  currentImage_ = std::move(nextImage);
  emit backgroundChanged(currentImage_);
  return true;
}

/**
 * @brief Go back to the previous image (wraps round); returns false if none
 * loaded.
 *
 * @return true if the previous image was set successfully, false otherwise
 */
bool Backgrounds::previous() {
  if (paths_.empty())
    return false;
  const size_t previousIdx =
      (currentIdx_ + paths_.size() - 1) % paths_.size();
  cv::Mat previousImage;
  if (!loadImage(paths_[previousIdx], previousImage))
    return false;
  currentIdx_ = previousIdx;
  currentImage_ = std::move(previousImage);
  emit backgroundChanged(currentImage_);
  return true;
}

/**
 * @brief How many images did we actually load?
 *
 * @return number of loaded images
 */
size_t Backgrounds::size() const noexcept { return paths_.size(); }

/**
 * @brief How many rows are in the current background?
 *
 * @return number of rows in the current image
 */
int Backgrounds::rows() const noexcept {
  if (currentImage_.empty())
    return 0;
  return currentImage_.rows;
}

/**
 * @brief How many columns are in the current background?
 *
 * @return number of columns in the current image
 */
int Backgrounds::cols() const noexcept {
  if (currentImage_.empty())
    return 0;
  return currentImage_.cols;
}
