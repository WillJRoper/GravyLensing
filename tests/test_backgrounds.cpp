#include <chrono>
#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>

#include "backgrounds.hpp"
#include "processing_geometry.hpp"

int main() {
  namespace fs = std::filesystem;
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path root = fs::temp_directory_path() / ("gravy-backgrounds-" + suffix);
  const fs::path first = root / "first";
  const fs::path second = root / "second";
  const fs::path large = root / "large";
  fs::create_directories(first);
  fs::create_directories(second);
  fs::create_directories(large);

  const cv::Mat red(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));
  const cv::Mat green(2, 2, CV_8UC3, cv::Scalar(0, 255, 0));
  for (int i = 0; i < 11; ++i)
    cv::imwrite((first / (std::to_string(i) + ".png")).string(), red);
  cv::imwrite((second / "custom.png").string(), green);
  cv::imwrite((large / "large.png").string(),
              cv::Mat(100, 300, CV_8UC3, cv::Scalar(255, 0, 0)));

  Backgrounds backgrounds(first.string());
  const bool loadedAll = Backgrounds::discoverableImageCount(first.string()) == 11 &&
                         backgrounds.load() && backgrounds.size() == 11;
  const bool navigated = backgrounds.next() && backgrounds.previous();
  const bool switched = backgrounds.setDirectory(second.string()) &&
                        backgrounds.size() == 1 &&
                        backgrounds.current().at<cv::Vec3b>(0, 0) ==
                            cv::Vec3b(0, 255, 0);
  const bool retained =
      !backgrounds.setDirectory((root / "missing").string()) &&
      backgrounds.size() == 1;
  Backgrounds resized(large.string(), 120, 40, "crop");
  const bool capped = resized.load() && resized.cols() == 120 &&
                      resized.rows() == 40;
  Backgrounds fitted(large.string(), 120, 120, "fit");
  const bool letterboxed =
      fitted.load() && fitted.cols() == 120 && fitted.rows() == 120 &&
      fitted.current().at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 0);
  Backgrounds stretched(large.string(), 80, 60, "stretch");
  const bool stretchedToSize =
      stretched.load() && stretched.cols() == 80 && stretched.rows() == 60;
  fs::remove(large / "large.png");
  const bool cacheReusable = resized.next() && resized.cols() == 120 &&
                             resized.rows() == 40;
  const bool geometry = calculationSize({1920, 1080}, 0.5f) ==
                            cv::Size(960, 540) &&
                        calculationSize({854, 480}, 0.35f) ==
                            cv::Size(299, 168) &&
                        resized.current().size() == cv::Size(120, 40);
  const cv::Mat retainedImage = resized.current().clone();
  Backgrounds::clearCache();
  const bool failedSwitchRetained =
      !resized.next() && !resized.current().empty() &&
      cv::norm(resized.current(), retainedImage, cv::NORM_INF) == 0.0;

  fs::remove_all(root);
  return loadedAll && navigated && switched && retained && capped &&
                 letterboxed && stretchedToSize && cacheReusable && geometry &&
                 failedSwitchRetained
             ? 0
             : 1;
}
