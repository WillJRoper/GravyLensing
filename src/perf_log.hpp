/**
 * @file perf_log.hpp
 *
 * Lightweight periodic profiling utilities for runtime pipeline stages.
 *
 * PerfLog is a thread-safe rolling-window logger that accumulates timing
 * samples and, every N samples, prints the average duration and effective
 * frames-per-second for one pipeline stage.  When the GRAVY_ENABLE_PROFILING
 * preprocessor flag is *not* defined, addSample() compiles to a no-op with
 * zero runtime overhead.
 *
 * Usage:
 *   static thread_local PerfLog perf("stage-name", 60);
 *   auto t0 = std::chrono::steady_clock::now();
 *   // ... work ...
 *   auto t1 = std::chrono::steady_clock::now();
 *   perf.addSample(std::chrono::duration<double, std::milli>(t1 - t0).count());
 *
 * Enable via CMake:
 *   cmake -B build -DENABLE_PROFILING=ON
 *
 * This file is part of GravyLensing, a real-time gravitational lensing
 * simulation.
 */
#pragma once

#include <chrono>
#include <iostream>
#include <string>

class PerfLog {
public:
  /// @param name        Label printed in log lines (e.g. "color-mask").
  /// @param reportEvery How many samples to accumulate before printing.
  explicit PerfLog(std::string name, int reportEvery = 60)
      : name_(std::move(name)), reportEvery_(reportEvery),
        windowStart_(std::chrono::steady_clock::now()) {}

  /// Accumulate a timing sample.  When enough samples have been collected
  /// the average ms and effective fps are printed to stdout.
  void addSample(double ms) {
#ifndef GRAVY_ENABLE_PROFILING
    (void)ms;
    return;
#else
    totalMs_ += ms;
    ++samples_;

    if (samples_ % reportEvery_ != 0)
      return;

    const auto now = std::chrono::steady_clock::now();
    const double wallMs = std::chrono::duration<double, std::milli>(
                              now - windowStart_)
                              .count();
    const double avgMs = totalMs_ / static_cast<double>(reportEvery_);
    const double fps = wallMs > 0.0 ? (1000.0 * reportEvery_) / wallMs : 0.0;

    std::cout << "[Perf] " << name_ << " avg=" << avgMs << "ms"
              << " fps=" << fps << " samples=" << reportEvery_ << "\n";

    totalMs_ = 0.0;
    windowStart_ = now;
#endif
  }

private:
  std::string name_;
  int reportEvery_;
  int samples_ = 0;
  double totalMs_ = 0.0;
  std::chrono::steady_clock::time_point windowStart_;
};
