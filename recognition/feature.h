#ifndef FEATURE_H
#define FEATURE_H

#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace multimedia {
  typedef std::map<std::string, std::vector<cv::KeyPoint> > KeyPointsMap;
  typedef std::map<std::string, cv::Mat_<float> > DescriptorsMap;
  
  namespace fs = boost::filesystem;
  
  class FeatureExtractor {
  public:
    FeatureExtractor();
    virtual ~FeatureExtractor() = default;
    void extract(const std::string& pathName);
    void extract(const std::string& pathName, cv::Mat& descriptors) const;
    void extractAll(const std::string& dirPathName);
    void read(const std::string& pathName, cv::Mat& descriptors,
              const std::string& keyName = "descriptors") const;
    void readAll(const std::string& dirPathName) const;
    void write(const std::string& pathName, const cv::Mat& descriptors,
               const std::string& keyName = "descriptors") const;
    void writeAll(const std::string& dirPathName, bool concat = true);
    const DescriptorsMap& getDescriptorsMap() const { return descriptorsMap; };
    int getSize() const { return descriptorsMap.size(); };
    void clear();
    
  private:
    FeatureExtractor(const FeatureExtractor&) = delete;
    FeatureExtractor& operator=(const FeatureExtractor&) = delete;

    std::unique_ptr<cv::ORB> feature;
    KeyPointsMap keypointsMap;
    DescriptorsMap descriptorsMap;
  };
}

#endif
