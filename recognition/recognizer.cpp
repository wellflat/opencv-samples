#include "recognizer.h"

namespace multimedia {
  Recognizer::Recognizer() {
  }

  void Recognizer::createVisualWords(const DescriptorsMap& descriptorsMap, 
                                     const std::string& vwFileName,
                                     int numVisualWords) {
    cv::BOWKMeansTrainer trainer(numVisualWords);
    DescriptorsMap::const_iterator it = descriptorsMap.begin();
    while(it != descriptorsMap.end()) {
      if(!it->second.empty()) {
        trainer.add(it->second);
      }
      ++it;
    }
    cv::Mat voc = trainer.cluster();
    cv::FileStorage storage(vwFileName, cv::FileStorage::WRITE);
    storage << "vocabulary" << voc;
    storage.release();
  }

  void Recognizer::train(const std::string& vocFileName,
                         const std::string& bowFileName) {
    /*
      cv::BOWImgDescriptorExtractor extractor(feature, new cv::FlannBasedMatcher());
      cv::Mat_<float> voc;
      cv::FileStorage storage(vocFileName, cv::FileStorage::READ);
      storage["vocablulary"] >> voc;
      extractor.setVocabulary(voc);

      cv::SVM svm;
    */
    //svm.train(data, response);
    //svm.save(bowFileName.c_str());
  }
}

