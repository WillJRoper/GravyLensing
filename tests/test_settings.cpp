#include <QCoreApplication>
#include <QSettings>
#include <QTemporaryDir>

#include "settings.hpp"

int main(int argc, char **argv) {
  QCoreApplication app(argc, argv);
  QTemporaryDir dir;
  if (!dir.isValid())
    return 1;

  QSettings stored(dir.filePath("settings.ini"), QSettings::IniFormat);
  AppSettings saved;
  saved.nthreads = 7;
  saved.personSensitivity = 73;
  saved.backgroundWidth = 1280;
  saved.backgroundHeight = 720;
  saved.backgroundFitMode = "fit";
  saved.cameraWidth = 1920;
  saved.cameraHeight = 1080;
  saved.lensEdgeSoftness = 2.3f;
  saved.colorMinObjectArea = 900;
  saved.colorPersistenceFrames = 12;
  saved.colorMaskSmooth = 0.3f;
  saved.lowerRes = 0.75f;
  saved.save(stored);

  AppSettings loaded;
  loaded.load(stored);
  if (loaded.nthreads != 7 || loaded.personSensitivity != 73 ||
      loaded.backgroundWidth != 1280 || loaded.backgroundHeight != 720 ||
      loaded.backgroundFitMode != "fit" || loaded.cameraWidth != 1920 ||
      loaded.cameraHeight != 1080 || loaded.lensEdgeSoftness != 2.3f ||
      loaded.colorMinObjectArea != 900 ||
      loaded.colorPersistenceFrames != 12 || loaded.colorMaskSmooth != 0.3f ||
      loaded.lowerRes != 0.75f)
    return 1;

  stored.setValue("automaticThreads", true);
  stored.setValue("nthreads", 2);
  loaded.load(stored);
  return !stored.contains("automaticThreads") &&
                 loaded.nthreads == AppSettings::defaultWorkerThreads()
             ? 0
             : 1;
}
