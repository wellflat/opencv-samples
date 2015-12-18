#include "fileio.hpp"
#include "train_test.hpp"

using namespace std;

int main(int argc, char** argv) {
  string dataBaseDir = "../data/";
  string protoFile = dataBaseDir + "deploy.prototxt";
  string modelFile = dataBaseDir + "bvlc_reference_caffenet.caffemodel";
  string trainListFile = dataBaseDir + "catnet/data/train.txt";
  //string trainListFile = dataBaseDir + "catnet/data/val.txt";
  string testListFile = dataBaseDir + "catnet/data/val.txt";
  //string testListFile = dataBaseDir + "catnet/data/train.txt";
  string svmModelFile = dataBaseDir + "cnn_svm_model.yml";
  // データセットの読み込み
  vector<cv::Mat> trainData, testData;
  vector<int> trainLabel, testLabel;
  
  enum Mode {
    TRAIN = 0,
    TEST = 1,
    PREDICT = 2
  };
  Mode mode = PREDICT;
  switch(mode) {
  case TRAIN:
    {
      cout << "extract feature and train SVM" << endl;
      cout << "training data load." << endl;
      loadImages(trainListFile, dataBaseDir + "catnet/", trainData, trainLabel);
      cout << trainData.size() << " data load complete." << endl;
      cv::Ptr<cv::ml::StatModel> clf = train(protoFile, modelFile, trainData, trainLabel);
      clf->save(svmModelFile);
      break;
    }
  case TEST:
  case PREDICT:
    {
      cv::Ptr<cv::dnn::Net> net = loadNet(protoFile, modelFile);
      cv::Ptr<cv::ml::SVM> clf = cv::Algorithm::load<cv::ml::SVM>(svmModelFile);
      cout << "C: " << clf->getC() << endl;
      if(mode == TEST) {
        cout << "test data load." << endl;
        loadImages(testListFile, dataBaseDir + "catnet/", testData, testLabel);
        cout << testData.size() << " data load complete." << endl;
        const cv::dnn::Blob input = cv::dnn::Blob(testData);
        cout << input.shape() << endl;
        net->setBlob(".data", input);
        net->forward();
        const cv::dnn::Blob blob = net->getBlob("fc7");
        const cv::Mat feature = blob.matRefConst();
        cout << "calculate error rate." << endl;
        cout << calculateError(clf, feature, cv::Mat(testLabel, false)) << endl;
      } else {
        string fileName = (argc > 1) ? argv[1] : "images/cat.jpg";
        cv::Mat image = cv::imread(fileName);
        if(image.empty()) {
          cerr << "can't read image: " << fileName << endl;
        }
        int cropSize = 224;
        cv::resize(image, image, cv::Size(cropSize, cropSize));
        const cv::dnn::Blob input = cv::dnn::Blob(image);
        cout << input.shape() << endl;
        net->setBlob(".data", input);
        net->forward();
        const cv::dnn::Blob blob = net->getBlob("fc7");
        const cv::Mat feature = blob.matRefConst();
        float pred = clf->predict(feature);
        cout << "predict label: " << pred << endl;
      }
      break;
    }
  default:
    break;
  }
  return 0;
}
