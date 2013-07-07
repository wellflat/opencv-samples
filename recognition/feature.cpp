#include "feature.h"

namespace multimedia {
  FeatureExtractor::FeatureExtractor() {
    feature.reset(new cv::ORB(500, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31));
  }

  void FeatureExtractor::extract(const std::string& pathName) {
    cv::Mat img = cv::imread(pathName);
    if(img.empty()) {
      throw std::runtime_error("cannot load image");
    }
    std::vector<cv::KeyPoint> keypoints;
    feature->detect(img, keypoints);
    keypointsMap.insert(KeyPointsMap::value_type(pathName, keypoints));
    cv::Mat_<float> descriptors;
    feature->compute(img, keypoints, descriptors);
    descriptorsMap.insert(DescriptorsMap::value_type(pathName, descriptors));
  }
  
  void FeatureExtractor::extract(const std::string& pathName,
                                 cv::Mat& descriptors) const {
    cv::Mat img = cv::imread(pathName);
    if(img.empty()) {
      throw std::runtime_error("cannot load image");
    }
    std::vector<cv::KeyPoint> keypoints;
    feature->detect(img, keypoints);
    feature->compute(img, keypoints, descriptors);
  }

  void FeatureExtractor::extractAll(const std::string& dirPathName) {
    fs::path dirPath = fs::path(dirPathName, fs::native);
    fs::directory_iterator end;
    for(fs::directory_iterator i(dirPath); i!=end; ++i) {
      /* i: basic_directory_entry */
      fs::path filePath = i->path();
      if(fs::is_directory(filePath)) continue;
      std::string pathName = i->string();
      cv::Mat img = cv::imread(pathName);
      if(img.empty()) {
        std::string message = "cannot load image: " + pathName;
        throw std::runtime_error(message);
      }
      std::vector<cv::KeyPoint> keypoints;
      feature->detect(img, keypoints);
      keypointsMap.insert(KeyPointsMap::value_type(pathName, keypoints));
      cv::Mat descriptors;
      feature->compute(img, keypoints, descriptors);
      descriptorsMap.insert(DescriptorsMap::value_type(pathName, descriptors));
    }
  }

  void FeatureExtractor::read(const std::string& pathName,
                              cv::Mat& descriptors,
                              const std::string& keyName) const {
    cv::FileStorage storage(pathName, cv::FileStorage::READ);
    storage[keyName] >> descriptors;
  }

  void FeatureExtractor::readAll(const std::string& dirPathName) const {
    fs::path dirPath = fs::path(dirPathName, fs::native);
    fs::directory_iterator end;
    for(fs::directory_iterator i(dirPath); i!=end; ++i) {
      /* i: basic_directory_entry */
      fs::path filePath = i->path();
      if(fs::is_directory(filePath)) continue;
      std::string pathName = i->string();
      // TODO: implements
    }
  }

  void FeatureExtractor::write(const std::string& pathName,
                               const cv::Mat& descriptors,
                               const std::string& keyName) const {
    cv::FileStorage storage(pathName, cv::FileStorage::WRITE);
    storage << keyName << descriptors;
    storage.release();
  }

  void FeatureExtractor::writeAll(const std::string& dirPathName,
                                  bool concat) {
    cv::FileStorage storage;
    fs::path path = fs::path(dirPathName, fs::native);
    DescriptorsMap::iterator it = descriptorsMap.begin();
    if(concat) {
      std::string yamlPathName = dirPathName + "/descriptors.yml";
      storage.open(yamlPathName, cv::FileStorage::WRITE);
    }
    while(it != descriptorsMap.end()) {
      std::string pathName = it->first;
      fs::path path(pathName);
      if(concat) {
        storage << "descriptors_" + path.stem() << it->second;
      } else {
        std::string yamlPathName = dirPathName + '/' + path.stem() + ".yml";
        storage.open(yamlPathName, cv::FileStorage::WRITE);
        storage << "descriptors_" + path.stem() << it->second;
        storage.release();
      }
      ++it;
    }
    if(concat) storage.release();
  }

  void FeatureExtractor::clear() {
    keypointsMap.clear();
    descriptorsMap.clear();
  }

}
