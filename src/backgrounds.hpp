/**
 * Backgrounds
 *
 * This class loads a set of up to 10 background images from a specified
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

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Backgrounds {
public:
  /// @param dir  path to folder containing your images
  explicit Backgrounds(const std::string &dir);

  /// Scan & load *all* images with “known” extensions.
  /// Returns false if directory doesn’t exist or no images found.
  bool load();

  /// Get the currently-selected image.
  const cv::Mat &current() const;

  /// Advance to the next image (wraps round); returns false if none loaded.
  bool next();

  /// Go back to the previous image (wraps round); returns false if none loaded.
  bool previous();

  /// Select image by zero-based index; returns false if idx out of range.
  bool setIndex(size_t idx);

  /// How many images did we actually load?
  size_t size() const noexcept;

  // Get the rows and cols of the current image
  int rows() const noexcept;
  int cols() const noexcept;

private:
  std::string dir_;
  std::vector<std::string> paths_;
  std::vector<cv::Mat> images_;
  size_t currentIdx_{0};

  // supported extensions (lower-case)
  static const std::vector<std::string> kImageExts;

  /// helper to load a single image by path
  static bool loadImage(const std::string &path, cv::Mat &out);
};
