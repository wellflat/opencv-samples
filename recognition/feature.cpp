#include "feature.h"

namespace multimedia {
  Feature::Feature() {
    feature.reset(new cv::ORB(500, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31));
  }

  void Feature::extract(const std::string& fileName) {
    fs::path filePath = fs::path(fileName, fs::native);
    cv::Mat img = cv::imread(fileName);
    if(img.empty()) {
      throw std::runtime_error("cannot load image");
    }
    std::vector<cv::KeyPoint> keypoints;
    feature->detect(img, keypoints);
    keypointsMap[filePath.stem()] = keypoints;
    cv::Mat descriptors;
    feature->compute(img, keypoints, descriptors);
    descriptorsMap[filePath.stem()] = descriptors;
  }
  
  void Feature::extract(const std::string& fileName,
                        cv::Mat& descriptors) const {
    cv::Mat img = cv::imread(fileName);
    if(img.empty()) {
      throw std::runtime_error("cannot load image");
    }
    std::vector<cv::KeyPoint> keypoints;
    feature->detect(img, keypoints);
    feature->compute(img, keypoints, descriptors);
  }

  void Feature::extractAll(const std::string& dirName) {
    fs::path dirPath = fs::path(dirName, fs::native);
    fs::directory_iterator end;
    for(fs::directory_iterator i(dirPath); i!=end; ++i) {
      /* i: basic_directory_entry */
      fs::path filePath = i->path();
      if(fs::is_directory(filePath)) continue;
      std::string fileName = i->string();
      cv::Mat img = cv::imread(fileName);
      if(img.empty()) {
        std::string message = "cannot load image: " + fileName;
        throw std::runtime_error(message);
      }
      std::vector<cv::KeyPoint> keypoints;
      feature->detect(img, keypoints);
      keypointsMap[filePath.stem()] = keypoints;
      cv::Mat descriptors;
      feature->compute(img, keypoints, descriptors);
      descriptorsMap[filePath.stem()] = descriptors;
    }
  }

  void Feature::read(const std::string& fileName,
                     cv::Mat& descriptors,
                     const std::string& keyName) const {
    cv::FileStorage storage(fileName, cv::FileStorage::READ);
    storage[keyName] >> descriptors;
    storage.release();
  }

  void Feature::readAll(const std::string& fileName,
                        const std::string& dirName) {
    cv::FileStorage storage(fileName, cv::FileStorage::READ);
    createKey(dirName);
    DescriptorsMap::const_iterator it = descriptorsMap.begin();
    while(it != descriptorsMap.end()) {
      cv::Mat_<float> descriptors;
      storage[it->first] >> descriptors;
      descriptorsMap[it->first] = descriptors;
      ++it;
    }
    storage.release();
  }

  void Feature::write(const std::string& fileName,
                      const cv::Mat& descriptors,
                      const std::string& keyName) const {
    cv::FileStorage storage(fileName, cv::FileStorage::WRITE);
    storage << keyName << (cv::Mat_<float>&)descriptors;
    storage.release();
  }

  void Feature::writeAll(const std::string& dirName, bool concat) {
    cv::FileStorage storage;
    fs::path path = fs::path(dirName, fs::native);
    DescriptorsMap::const_iterator it = descriptorsMap.begin();
    if(concat) {
      std::string yamlName = dirName + "/descriptors.yml";
      storage.open(yamlName, cv::FileStorage::WRITE);
    }
    while(it != descriptorsMap.end()) {
      if(concat) {
        storage << it->first << it->second;
      } else {
        std::string yamlName = dirName + '/' + it->first + ".yml";
        storage.open(yamlName, cv::FileStorage::WRITE);
        storage << path.stem() << it->second;
        storage.release();
      }
      ++it;
    }
    if(concat) storage.release();
  }

  int Feature::getDescriptorCount() const {
    int n = 0;
    DescriptorsMap::const_iterator it = descriptorsMap.begin();
    while(it != descriptorsMap.end()) {
      n += it->second.rows;
      ++it;
    }
    return n;
  }
  
  void Feature::clear() {
    keypointsMap.clear();
    descriptorsMap.clear();
  }

  /*
   * private methods
   */
  void Feature::createKey(const std::string& dirName){
    fs::path dirPath = fs::path(dirName, fs::native);
    fs::directory_iterator end;
    for(fs::directory_iterator i(dirPath); i!=end; ++i) {
      fs::path filePath = i->path();
      if(fs::is_directory(filePath)) continue;
      keypointsMap[filePath.stem()];
      descriptorsMap[filePath.stem()];
    }
  }

}
