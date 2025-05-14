/**
 * @file lensing_worker.hpp
 *
 * This defines the worker class used to calculate the lensing effect. A
 * new lens is calculated for every new mask.
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
#include <QObject>

// External includes
#include <fftw3.h>
#include <opencv2/opencv.hpp>

// Local includes

class LensingWorker : public QObject {
  Q_OBJECT

public:
  // ================== Member Variable Declarations ==================

  // ================== Member Function Prototypes ==================

  // Constructor
  LensingWorker(float strength = 1.0f, float softening = 30.0f,
                int padFactor = 2, int nthreads = 1, float lowerRes = 1.0f,
                bool distortInside = false);

  // Destructor
  ~LensingWorker();

public Q_SLOTS:

  // Calculate a new lensing effect when there is a new mask
  void onMask(const cv::Mat &mask);

  // Update the geometry when the background changes
  void onBackgroundChange(const cv::Mat &background);

signals:

  // Signal to indicate that lensing is ready
  void lensedReady(const cv::Mat &lensedImage);

  // Signal to indicate that lensing is ready with a mask
  void lensingError(const std::string &error);

private:
  // ================== Member Variable Declarations ==================

  // The base level lens strength
  float strength_;

  // The softening radius for the lens
  float softening_;

  // The padding factor for the FFT
  int padFactor_;

  // The number of space threads (excluding Qt threads)
  int nthreads_;

  // The dimensions
  int width_, height_;
  int padWidth_, padHeight_; // padded dimensions for FFTs

  // The lower resolution factor for the lensing effect. The resolution at which
  // the lensing effect is calculed will be this much smaller than the
  // background resolution.
  float lowerRes_;

  // The current background image
  cv::Mat currentBackground_;

  // The latest lensed image
  cv::Mat latestLensed_;

  // FFTW buffers and plans for kernel transforms
  float *kernelX_ = nullptr;
  float *kernelY_ = nullptr;
  fftwf_complex *Kx_ft_ = nullptr;
  fftwf_complex *Ky_ft_ = nullptr;
  fftwf_plan planKx_ = nullptr;
  fftwf_plan planKy_ = nullptr;

  // FFT buffers/plans for mask and deflection
  float *maskBuf_ = nullptr;        // size padH*padW
  fftwf_complex *maskFT_ = nullptr; // size padH*(padW/2+1)
  fftwf_complex *defFT_ = nullptr;
  float *defBuf_ = nullptr;

  // FFTW plans for mask and deflection
  fftwf_plan planMask_ = nullptr;
  fftwf_plan planDef_ = nullptr;

  // Deflection maps used during applyLensing
  cv::Mat mapX_, mapY_;

  // Some matrices we can reuse during lens application
  cv::Mat padded_;
  cv::Mat paddedF_;

  // Are we distorting inside the mask?
  bool distortInside_;

  // ================== Member Function Prototypes ==================

  // Precompute and cache the FFTs of the deflection kernels.
  void buildKernels();

  // Apply gravitational lensing to a static background using a binary mask.
  void applyLensing(const cv::Mat &mask);

  // Allocate the FFTW plans and buffers for the kernels
  void allocateFFTWKernels();

  // Free the FFTW plans and buffers for the kernels
  void freeFFTWKernels();

  // Allocate the FFTW plans and buffers for the deflections
  void allocateFFTWDeflections();

  // Free the FFTW plans and buffers for the deflections
  void freeFFTWDeflections();

  // Update the geometry of the lensing effect
  void updateGeometry(int width, int height);
};
