#include <iostream>
#include <opencv2/datasets/or_mnist.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

int main(int argc, char** argv) {
  /* ../data/t10k-images.idx3-ubyte
     ../data/t10k-labels.idx1-ubyte
     ../data/train-images.idx3-ubyte
     ../data/train-labels.idx1-ubyte */
  string filePath = "../data/";
  cv::Ptr<cv::datasets::OR_mnist> dataset = cv::datasets::OR_mnist::create();
  dataset->load(filePath);
  const vector<cv::Ptr<cv::datasets::Object> >& trainData = dataset->getTrain();
  const vector<cv::Ptr<cv::datasets::Object> >& testData = dataset->getTest();
  cout << "train data size: " << trainData.size() << endl;  // 60000
  cout << "test data size: " << testData.size() << endl;  // 10000
  // struct OR_mnistObj : public Object
  // {
  //     char label; // 0..9
  //     Mat image; // [28][28]
  // };
  // 一つめのサンプルを取得
  cv::datasets::OR_mnistObj* example = static_cast<cv::datasets::OR_mnistObj*>(trainData[0].get());
  cout << "label: " << static_cast<int>(example->label) << endl;  // label: 5
  cout << example->image.size() << endl;  // [28 x 28]
  cv::imwrite("train0.png", example->image);
  return 0;
}
                                 
