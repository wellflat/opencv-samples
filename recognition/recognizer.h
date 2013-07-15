#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <vector>
#include <map>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace multimedia {
  typedef std::map<std::string, std::vector<cv::KeyPoint> > KeyPointsMap;
  typedef std::map<std::string, cv::Mat_<float> > DescriptorsMap;
  
  namespace fs = boost::filesystem;
  
  class Recognizer {
  public:
    Recognizer();
    virtual ~Recognizer() = default;

    void train(const std::string& vwFileName,
               const std::string& bowFileName);
    void createVisualWords(const DescriptorsMap& descriptorsMap,
                           const std::string& vwFileName,
                           int numVisualWords);
    
  private:
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;
  };
}

#endif

