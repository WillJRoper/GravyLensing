#include <chrono>
#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>

#include "backgrounds.hpp"

int main() {
  namespace fs = std::filesystem;
  const auto suffix = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path root = fs::temp_directory_path() / ("gravy-backgrounds-" + suffix);
  const fs::path first = root / "first";
  const fs::path second = root / "second";
  fs::create_directories(first);
  fs::create_directories(second);

  const cv::Mat red(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));
  const cv::Mat green(2, 2, CV_8UC3, cv::Scalar(0, 255, 0));
  for (int i = 0; i < 11; ++i)
    cv::imwrite((first / (std::to_string(i) + ".png")).string(), red);
  cv::imwrite((second / "custom.png").string(), green);

  Backgrounds backgrounds(first.string());
  const bool loadedAll = backgrounds.load() && backgrounds.size() == 11;
  const bool navigated = backgrounds.next() && backgrounds.previous();
  const bool switched = backgrounds.setDirectory(second.string()) &&
                        backgrounds.size() == 1 &&
                        backgrounds.current().at<cv::Vec3b>(0, 0) ==
                            cv::Vec3b(0, 255, 0);
  const bool retained =
      !backgrounds.setDirectory((root / "missing").string()) &&
      backgrounds.size() == 1;

  fs::remove_all(root);
  return loadedAll && navigated && switched && retained ? 0 : 1;
}
