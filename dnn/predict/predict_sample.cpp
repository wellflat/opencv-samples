#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

int main(int argc, char** argv) {
  string dataBaseDir = "../data/";
  // ImageNet Caffeリファレンスモデル
  string protoTxtFile = dataBaseDir + "bvlc_reference_caffenet/deploy.prototxt";
  string caffeModelFile = dataBaseDir + "bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
  string wordsFile = dataBaseDir + "bvlc_reference_caffenet/synset_words.txt";
  // 動作確認用の画像ファイル
  string imageFile = (argc > 1) ? argv[1] : dataBaseDir + "images/cat.jpg";
  // Caffeモデルの読み込み
  cv::Ptr<cv::dnn::Importer> importer;
  try {
    importer = cv::dnn::createCaffeImporter(protoTxtFile, caffeModelFile);
  } catch(const cv::Exception& e) {
    cerr << e.msg << endl;
    exit(-1);
  }
  cv::dnn::Net net;
  importer->populateNet(net);
  importer.release();
  // 画像ファイルの読み込み
  cv::Mat img = cv::imread(imageFile);
  if(img.empty()) {
    cerr << "can't read image: " << imageFile << endl;
    exit(-1);
  }
  try {
    // 入力画像をリサイズ
    int cropSize = 224;
    cv::resize(img, img, cv::Size(cropSize, cropSize));
    // Caffeで扱うBlob形式に変換 (実体はcv::Matのラッパークラス)
    const cv::dnn::Blob inputBlob = cv::dnn::Blob(img);
    // 入力層に画像を入力
    net.setBlob(".data", inputBlob);
    // フォワードパス(順伝播)の計算
    net.forward();
    // 出力層(Softmax)の出力を取得, ここに予測結果が格納されている
    const cv::dnn::Blob prob = net.getBlob("prob");
    // Blobオブジェクト内部のMatオブジェクトへの参照を取得
    // ImageNet 1000クラス毎の確率が格納された1x1000の行列(ベクトル)
    const cv::Mat probMat = prob.matRefConst();
    // 確率(信頼度)の高い順にソートして、上位5つのインデックスを取得
    cv::Mat sorted(probMat.rows, probMat.cols, CV_32F);
    cv::sortIdx(probMat, sorted, CV_SORT_EVERY_ROW|CV_SORT_DESCENDING);
    cv::Mat topk = sorted(cv::Rect(0, 0, 5, 1));
    // カテゴリ名のリストファイル(synset_words.txt)を読み込み
    // データ例: categoryList[951] = "lemon";
    vector<string> categoryList;
    string category;
    ifstream fs(wordsFile.c_str());
    if(!fs.is_open()) {
      cerr << "can't read file" << endl;
      exit(-1);
    }
    while(getline(fs, category)) {
      if(category.length()) {
        categoryList.push_back(category.substr(category.find(' ') + 1));
      }
    }
    fs.close();
    // 予測したカテゴリと確率(信頼度)を出力
    cv::Mat_<int>::const_iterator it = topk.begin<int>();
    while(it != topk.end<int>()) {
      cout << categoryList[*it] << " : " << probMat.at<float>(*it) * 100 << " %" << endl;
      ++it;
    }
  } catch(const cv::Exception& e) {
    cerr << e.msg << endl;
  }
  return 0;
}
