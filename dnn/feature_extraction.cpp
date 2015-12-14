#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

void loadImages(string listFileName, vector<cv::Mat>& trainData, vector<int>& trainLabel,
                string rootPath) {
  // row: filepath label\n
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
    current = found + 1;
    label = stoi(string(line, current, line.size() - current));
    trainData.push_back(image);
    trainLabel.push_back(label);
  }
  fs.close();
}

const cv::Mat extractFeature(string protoTxtFile, string modelFile, vector<cv::Mat>& images) {
  cv::Ptr<cv::dnn::Importer> importer;
  try {
    importer = cv::dnn::createCaffeImporter(protoTxtFile, modelFile);
  } catch(const cv::Exception& e) {
    cerr << e.msg << endl;
    exit(-1);
  }
  cv::dnn::Net net;
  importer->populateNet(net);
  importer.release();
  
  try {
    // int cropSize = 224;
    // cv::resize(img, img, cv::Size(cropSize, cropSize));
    const cv::dnn::Blob input = cv::dnn::Blob(images);
    net.setBlob(".data", input);
    net.forward();
    // 全結合層 fc7(InnerProduct)の出力を特徴量として抽出
    const cv::dnn::Blob blob = net.getBlob("fc7");
    const cv::Mat feature = blob.matRefConst();
    return feature;
  } catch(const cv::Exception& e) {
    cerr << e.msg << endl;
  }
  
}

cv::Ptr<cv::ml::SVM> train(const cv::Mat& X, cv::Mat& y) {
  cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(X, cv::ml::ROW_SAMPLE, y);
  cv::Ptr<cv::ml::SVM> clf = cv::ml::SVM::create();
  clf->setType(cv::ml::SVM::C_SVC);
  clf->setKernel(cv::ml::SVM::LINEAR);
  clf->setC(1000);
  clf->train(data);
  return clf;
}

int main(int argc, char** argv) {
  string protoTxtFile = "bvlc_reference_caffenet/deploy.prototxt";
  string caffeModelFile = "bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
  //string trainListFile = "catnet/data/train.txt";
  string trainListFile = "catnet/data/val.txt";
  string testListFile = "catnet/data/val.txt";

  // データセットの読み込み
  vector<cv::Mat> trainData, testData;
  vector<int> vtrainLabel, vtestLabel;
  cv::Mat trainLabel, testLabel;
  cout << "train data load." << endl;
  loadImages(trainListFile, trainData, vtrainLabel, "catnet/../../");
  cout << trainData.size() << " data load complete." << endl;
  cout << "test data load." << endl;
  loadImages(testListFile, testData, vtestLabel, "catnet/../../");
  cout << testData.size() << " data load complete." << endl;
  //cout << "random shuffle train data." << endl;
  //random_shuffle(trainData.begin(), trainData.end());
  trainLabel = cv::Mat(vtrainLabel, true);
  testLabel = cv::Mat(vtestLabel, true);
  
  //for(auto it = begin(trainData); it != end(trainData); ++it) {
    //cout << (*it).size() << endl;
  //}

  cout << "extract feature." << endl;
  const cv::Mat feature = extractFeature(protoTxtFile, caffeModelFile, trainData);
  cout << feature.size() << " extract feature complete." << endl;
  
  cout << "train model start." << endl;
  cv::Ptr<cv::ml::SVM> clf = train(feature, trainLabel);
  cout << "train model complete." << endl;
  clf->save("cnn_svm_model.yml");
  return 0;
}
