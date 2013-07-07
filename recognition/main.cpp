#include <iostream>
#include <stdexcept>
#include "feature.h"

/* test code */
int main(int argc, char** argv) {
  using namespace std;
  try {
    multimedia::FeatureExtractor extractor;
    string basePath = "../dstfiles/kitten/";
    string pathName = basePath + "frame100.jpg";
    string dirPathName = basePath;
    string yamlPathName = basePath + "features/frame100.yml";
    cv::Mat descriptors, descriptors2;
    extractor.extract(pathName);
    extractor.extract(pathName, descriptors);
    extractor.write("test.yml", descriptors);
    //extractor.extractAll(dirPathName);
    extractor.read("test.yml", descriptors2);
    //cout << descriptors2 << endl;
    cout << "size: " << extractor.getSize() << endl;
    //extractor.writeAll("../dstfiles/kitten/features", true);
    multimedia::DescriptorsMap descriptorsMap = extractor.getDescriptorsMap();
    multimedia::DescriptorsMap::iterator it = descriptorsMap.begin();
    while(it != descriptorsMap.end()) {
      cout << it->first << endl; // path
      ++it;
    }

  } catch(const std::exception& e) {
    cerr << e.what() << endl;
    return -1;
  }
  return 0;
}
