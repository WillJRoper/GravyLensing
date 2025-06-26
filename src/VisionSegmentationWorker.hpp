#pragma once
#include "PersonSegmenter.hpp"
#include <QObject>
#include <opencv2/opencv.hpp>

class VisionSegmentationWorker : public QObject {
  Q_OBJECT
public:
  VisionSegmentationWorker(QObject *parent = nullptr);
public slots:
  void onFrame(const cv::Mat &frame);
signals:
  void maskReady(const cv::Mat &mask);

private:
  PersonSegmenter _seg;
};
