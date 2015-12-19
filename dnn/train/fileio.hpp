#ifndef __FILEIO_HPP__
#define __FILEIO_HPP__

#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

void loadImages(string listFileName, string rootPath,
                vector<cv::Mat>& images, vector<int>& labels, /* out parameters */
                bool isResize=false, int cropSize=224);

const vector<string> loadSynsetWords(string fileName);

cv::Ptr<cv::dnn::Net> loadNet(string protoFile, string modelFile);

bool saveMat(string fileName, string key, const cv::Mat& mat);

const cv::Mat loadMat(string fileName, string key);

#endif

