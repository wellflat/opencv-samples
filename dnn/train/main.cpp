#include "fileio.hpp"
#include "train_test.hpp"

using namespace std;

int main(int argc, char** argv) {
  string dataBaseDir = "../data/";
  string protoFile = dataBaseDir + "bvlc_reference_caffenet/deploy.prototxt";
  string modelFile = dataBaseDir + "bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
  string trainListFile = dataBaseDir + "catnet/data/val.txt";
  string testListFile = dataBaseDir + "catnet/data/val.txt";
  string svmModelFile = dataBaseDir + "model.yml";
  vector<cv::Mat> trainData, testData;
  vector<int> trainLabel, testLabel;
  enum Mode { PREDICT = 0, TRAIN = 1, TEST = 2};
  const cv::String keys =
    "{h help |               |print this message       }"
    "{m mode |0              |0:predict/1:train/2:test }"
    "{@image |images/cat.jpg |image file               }"
    ;
  cv::CommandLineParser parser(argc, argv, keys);
  if(parser.has("h")) {
    parser.printMessage();
    return 0;
  }
  int mode = parser.get<int>("mode");
  switch(mode) {
  case TRAIN:
    {
      cout << "extract feature and train SVM" << endl;
      cout << "training data load." << endl;
      bool isResize = false;
      loadImages(trainListFile, dataBaseDir + "catnet/data/", trainData, trainLabel, isResize);
      cout << trainData.size() << " data load complete." << endl;
      cv::Ptr<cv::ml::StatModel> clf = train(protoFile, modelFile, trainData, trainLabel);
      cout << "save model: " << svmModelFile << endl;
      clf->save(svmModelFile);
      break;
    }
  case TEST:
    {
      cv::Ptr<cv::dnn::Net> net = loadNet(protoFile, modelFile);
      cv::Ptr<cv::ml::SVM> clf = cv::Algorithm::load<cv::ml::SVM>(svmModelFile);
      cout << "Cost: " << clf->getC() << endl;
      cout << "test data load." << endl;
      loadImages(testListFile, dataBaseDir + "catnet/data/", testData, testLabel);
      cout << testData.size() << " data load complete." << endl;
      const cv::dnn::Blob input = cv::dnn::Blob(testData);
      cout << input.shape() << endl;
      net->setBlob(".data", input);
      net->forward();
      const cv::dnn::Blob blob = net->getBlob("fc7");
      const cv::Mat feature = blob.matRefConst();
      cout << "calculate error rate." << endl;
      cout << calculateError(clf, feature, cv::Mat(testLabel, false)) << endl;
      break;
    }
  case PREDICT:
    {
      cv::Ptr<cv::dnn::Net> net = loadNet(protoFile, modelFile);
      cv::Ptr<cv::ml::SVM> clf = cv::Algorithm::load<cv::ml::SVM>(svmModelFile);
      string fileName = parser.get<string>(0);
      cv::Mat image = cv::imread(fileName);
      if(image.empty()) {
        cerr << "can't read image: " << fileName << endl;
        exit(-1);
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
      break;
    }
  default:
    break;
  }
  return 0;
}
