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
  
  class Feature {
  public:
    Feature();
    virtual ~Feature() = default;
    void extract(const std::string& fileName);
    void extract(const std::string& fileName, cv::Mat& descriptors) const;
    void extractAll(const std::string& dirName);
    void read(const std::string& fileName, cv::Mat& descriptors,
              const std::string& keyName = "descriptors") const;
    void readAll(const std::string& fileName,
                 const std::string& dirName);
    void write(const std::string& fileName,
               const cv::Mat& descriptors,
               const std::string& keyName = "descriptors") const;
    void writeAll(const std::string& dirName, bool concat = true);
    void createVisualWords(const std::string& fileName,
                           int numVisualWords);
    void train(const std::string& vocFileName,
               const std::string& bowFileName);
               
    const DescriptorsMap& getDescriptorsMap() const { return descriptorsMap; };
    int getSize() const { return descriptorsMap.size(); };
    void clear();
    
  private:
    Feature(const Feature&) = delete;
    Feature& operator=(const Feature&) = delete;

    int getDescriptorCount() const;
    void createKey(const std::string& dirName);

    std::unique_ptr<cv::ORB> feature;
    KeyPointsMap keypointsMap;
    DescriptorsMap descriptorsMap;
  };
}

#endif
