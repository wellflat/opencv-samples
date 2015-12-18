#include "fileio.hpp"

/*
 * list file format
 * imagefilepath label\nimagefilepath label\n ...
 */
void loadImages(string listFileName, string rootPath,
                vector<cv::Mat>& images, vector<int>& labels,
                bool isResize, int cropSize) {
  ifstream fs(listFileName.c_str());
  if(!fs.is_open()) {
    cerr << "can't read list file" << endl;
    exit(-1);
  }
  string line, filePath;
  int label;
  while(getline(fs, line)) {
    int current = 0, found;
    found = line.find(' ', current);
    filePath = string(line, current, found - current);
    cv::Mat image = cv::imread(rootPath + filePath);
    if(image.empty()) continue;
    if(isResize) {
      cv::resize(image, image, cv::Size(cropSize, cropSize));
    }
    current = found + 1;
    label = atoi(string(line, current, line.size() - current).c_str());
    images.push_back(image);
    labels.push_back(label);
  }
  fs.close();
}

const vector<string> loadSynsetWords(string fileName) {
  vector<string> labels;
  string label;
  ifstream fs(fileName.c_str());
  if(!fs.is_open()) {
    cerr << "can't read file" << endl;
    exit(-1);
  }
  while(getline(fs, label)) {
    if(label.length()) {
      labels.push_back(label.substr(label.find(' ') + 1));
    }
  }
  fs.close();
  return labels;  // expects NRVO
}

cv::Ptr<cv::dnn::Net> loadNet(string protoFile, string modelFile) {
  cv::Ptr<cv::dnn::Importer> importer;
  importer = cv::dnn::createCaffeImporter(protoFile, modelFile);
  cv::Ptr<cv::dnn::Net> net = new cv::dnn::Net();
  importer->populateNet(*net);
  importer.release();
  return net;
}

void saveMat(string fileName, string key, const cv::Mat& mat) {
  cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
  cv::internal::WriteStructContext ws(fs, key, CV_NODE_SEQ);
  cv::write(fs, "", mat);
}

const cv::Mat loadMat(string fileName, string key) {
  // cv::FileStorage fs(fileName, cv::FileStorage::READ);
  // cv::internal::WriteStructContext ws(fs, key, CV_NODE_SEQ);
  cv::Mat mat;
  // cv::read(fs, "", mat);
  return mat;  // expects NRVO
}
