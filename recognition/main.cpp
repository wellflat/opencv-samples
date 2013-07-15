#include <iostream>
#include <stdexcept>
#include "feature.h"

/* test code */
int main(int argc, char** argv) {
  using namespace std;
  try {
    multimedia::Feature feature;
    string basePath = "../dstfiles/kitten/";
    string fileName = basePath + "frame100.jpg";
    string yamlName = basePath + "features/frame100.yml";
    cv::Mat descriptors, descriptors2;
    //feature.extract(fileName);
    //feature.extract(fileName, descriptors);
    //cout << descriptors << endl;
    //feature.extractAll(basePath);
    //feature.write("test.yml", descriptors);
    // feature.read("test.yml", descriptors2);
    // cout << "rows: " << descriptors2.rows << ", cols: "
    //       << descriptors2.cols << endl;
    
    //feature.writeAll(basePath + "features", true);
    feature.readAll(basePath + "features/descriptors.yml", basePath);
    cout << "image count: " << feature.getSize() << endl;
    multimedia::DescriptorsMap descriptorsMap = feature.getDescriptorsMap();
    multimedia::DescriptorsMap::iterator it = descriptorsMap.begin();
    // while(it != descriptorsMap.end()) {
    //   cout << it->first << endl; // path
    //   ++it;
    // }
    int n = 100;
    int descriptorsCount = feature.createVisualWords("voc.yml", n);
    cout << "descriptors count: " << descriptorsCount << endl;
    cout << "visual word count: " <<  n << endl;

  } catch(const std::exception& e) {
    cerr << e.what() << endl;
    return -1;
  }
  return 0;
}
