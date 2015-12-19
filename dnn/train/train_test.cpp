#include "train_test.hpp"

cv::Ptr<cv::ml::StatModel>
train(const string& protoFile, const string& modelFile,
      const vector<cv::Mat>& trainData, const vector<int>& trainLabel,
      const string& type, int kFold) {
  cv::Ptr<cv::dnn::Net> net = loadNet(protoFile, modelFile);
  const cv::dnn::Blob input = cv::dnn::Blob(trainData);
  cout << input.shape() << endl;
  net->setBlob(".data", input);
  net->forward();
  // 全結合層 fc7(InnerProduct)の出力を特徴量として抽出
  cout << "extract feature" << endl;
  const cv::dnn::Blob blob = net->getBlob("fc7");
  net.release();
  const cv::Mat feature = blob.matRefConst();
  cout << feature.size() << endl;
  cout << "train model" << endl;
  // 線形SVM or ロジスティック回帰で学習
  if(type == "SVM") {
    cv::Ptr<cv::ml::TrainData> data =
      cv::ml::TrainData::create(feature, cv::ml::ROW_SAMPLE, cv::Mat(trainLabel, false));
    cv::Ptr<cv::ml::SVM> clf = cv::ml::SVM::create();
    clf->setType(cv::ml::SVM::C_SVC);
    clf->setKernel(cv::ml::SVM::LINEAR);
    clf->trainAuto(data, kFold); // グリッドサーチ + 交差検証
    return clf;
  } else {
    cv::Mat tmp = cv::Mat(trainLabel, false);
    cv::Mat trainLabel32F;
    tmp.convertTo(trainLabel32F, CV_32F);
      cv::Ptr<cv::ml::TrainData> data =
        cv::ml::TrainData::create(feature, cv::ml::ROW_SAMPLE, trainLabel32F);
    cv::Ptr<cv::ml::LogisticRegression> clf =
      cv::ml::LogisticRegression::create();
    clf->setLearningRate(0.001);
    clf->setIterations(10000);
    clf->setRegularization(cv::ml::LogisticRegression::REG_L2);
    clf->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
    clf->setMiniBatchSize(20);
    clf->train(data);
    return clf;
  }
}

float calculateError(const cv::Ptr<cv::ml::StatModel> clf,
                     const cv::Mat& testData, const cv::Mat& testLabel,
                     bool verbose) {
  cv::Mat testLabel32F;
  testLabel.convertTo(testLabel32F, CV_32F);
  cv::Ptr<cv::ml::TrainData> data =
    cv::ml::TrainData::create(testData, cv::ml::ROW_SAMPLE, testLabel32F);
  cv::Mat res;
  float err = clf->calcError(data, false, res);
  if(verbose) {
    for(int n=0; n<testLabel.rows; ++n) {
      cerr << testLabel32F.at<float>(n, 0) << " : " << res.at<float>(n, 0) << endl;
    }
  }
  return err;
}
