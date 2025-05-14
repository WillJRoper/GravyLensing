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

// Local includes
#include "backgrounds.hpp"

namespace fs = std::filesystem;

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
Backgrounds *initBackgrounds(const std::string &dir) {
  Backgrounds *backgrounds = new Backgrounds(dir);
  if (!backgrounds->load()) {
    qFatal("No images found in directory: %s", dir.c_str());
    return nullptr;
  }
  return backgrounds;
}

/**
 * @brief Backgrounds constructor
 *
 * @param dir path to folder containing your images
 */
Backgrounds::Backgrounds(const std::string &dir) : dir_(dir) {}

/**
 * @brief Scan & load *all* images with “known” extensions.
 *
 * @return false if directory doesn’t exist or no images found.
 */
bool Backgrounds::load() {
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
      paths_.push_back(entry.path().string());
    }
  }

  if (paths_.empty())
    return false;

  // sort so numbering is stable
  std::sort(paths_.begin(), paths_.end());

  // load into memory
  images_.reserve(paths_.size());
  for (auto const &p : paths_) {
    cv::Mat img;
    if (loadImage(p, img)) {
      images_.push_back(std::move(img));
    }
  }

  // Raise an error if there are more than 10 images
  if (images_.size() > 10) {
    std::cerr << "Warning: More than 10 images loaded, only the first 10 are "
                 "accessible."
              << std::endl;
    images_.resize(10);
  }

  return !images_.empty();
}

/**
 * @brief Load a single image by path
 *
 * @param path path to the image
 * @param out output cv::Mat
 *
 * @return true if the image was loaded successfully, false otherwise
 */
bool Backgrounds::loadImage(const std::string &path, cv::Mat &out) {
  out = cv::imread(path, cv::IMREAD_COLOR);
  return !out.empty();
}

/**
 * @brief Get the currently-selected image.
 *
 * @return reference to the current image
 */
const cv::Mat &Backgrounds::current() const { return images_[currentIdx_]; }

/**
 * @brief Advance to the next image (wraps round); returns false if none loaded.
 *
 * @return true if the next image was set successfully, false otherwise
 */
bool Backgrounds::next() {
  if (images_.empty())
    return false;
  currentIdx_ = (currentIdx_ + 1) % images_.size();
  emit backgroundChanged(images_[currentIdx_]);
  return true;
}

/**
 * @brief Go back to the previous image (wraps round); returns false if none
 * loaded.
 *
 * @return true if the previous image was set successfully, false otherwise
 */
bool Backgrounds::previous() {
  if (images_.empty())
    return false;
  currentIdx_ = (currentIdx_ + images_.size() - 1) % images_.size();
  return true;
}

/**
 * @brief Select image by zero-based index; returns false if idx out of range.
 *
 * @param idx index of the image to set
 * @return true if the image was set successfully, false otherwise
 */
bool Backgrounds::setIndex(size_t idx) {
  if (idx >= images_.size())
    return false;
  currentIdx_ = idx;
  emit backgroundChanged(images_[currentIdx_]);
  return true;
}

/**
 * @brief How many images did we actually load?
 *
 * @return number of loaded images
 */
size_t Backgrounds::size() const noexcept { return images_.size(); }

/**
 * @brief How many rows are in the current background?
 *
 * @return number of rows in the current image
 */
int Backgrounds::rows() const noexcept {
  if (images_.empty())
    return 0;
  return images_[currentIdx_].rows;
}

/**
 * @brief How many columns are in the current background?
 *
 * @return number of columns in the current image
 */
int Backgrounds::cols() const noexcept {
  if (images_.empty())
    return 0;
  return images_[currentIdx_].cols;
}
