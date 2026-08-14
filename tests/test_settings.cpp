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
  saved.automaticThreads = false;
  saved.nthreads = 7;
  saved.personSensitivity = 73;
  saved.save(stored);

  AppSettings loaded;
  loaded.load(stored);
  if (loaded.automaticThreads || loaded.nthreads != 7 ||
      loaded.personSensitivity != 73)
    return 1;

  stored.setValue("automaticThreads", true);
  stored.setValue("nthreads", 2);
  loaded.load(stored);
  return loaded.automaticThreads &&
                 loaded.nthreads == AppSettings::recommendedThreads()
             ? 0
             : 1;
}
