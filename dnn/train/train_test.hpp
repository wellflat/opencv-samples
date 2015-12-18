#ifndef __TRAINTEST_HPP__
#define __TRAINTEST_HPP__

#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include "fileio.hpp"

using namespace std;

cv::Ptr<cv::ml::StatModel>
train(const string& protoFile, const string& modelFile,
      const vector<cv::Mat>& trainData, const vector<int>& trainLabel,
      const string& type="SVM", int kFold=5);

float
calculateError(const cv::Ptr<cv::ml::StatModel> clf,
               const cv::Mat& testData, const cv::Mat& testLabel,
               bool verbose=false);

#endif
