#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

int main(int argc, char** argv) {
  // ImageNet Caffeリファレンスモデル
  string protoTxtFile = "bvlc_reference_caffenet/deploy.prototxt";
  string caffeModelFile = "bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
  string imageFile = (argc > 1) ? argv[1] : "images/cat.jpg";
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
  // テスト用の入力画像ファイルの読み込み
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
    // フォワードパス(順伝播)実行
    net.forward();
    // 出力層(Softmax)の出力を取得, ここに予測結果が格納されている
    const cv::dnn::Blob prob = net.getBlob("prob");
    // Blobオブジェクト内部のMatオブジェクトへの参照を取得
    // ImageNet 1000クラスの確率(32bits浮動小数点値)が格納された1x1000の行列(ベクトル)
    const cv::Mat probMat = prob.matRefConst();
    // 確率(信頼度)の高い順にソートして、上位5つのインデックスを取得
    cv::Mat sorted(probMat.rows, probMat.cols, CV_32F);
    cv::sortIdx(probMat, sorted, CV_SORT_EVERY_ROW|CV_SORT_DESCENDING);
    cv::Mat topk = sorted(cv::Rect(0, 0, 5, 1));
    // カテゴリ名のリストファイル(synset_words.txt)を読み込み
    // データ例: classNames[951] = "lemon";
    vector<string> classNames;
    string className;
    ifstream fp("synset_words.txt");
    if(!fp.is_open()) {
      cerr << "can't read file" << endl;
      exit(-1);
    }
    while(!fp.eof()) {
      std::getline(fp, className);
      if(className.length()) {
        classNames.push_back(className.substr(className.find(' ') + 1));
      }
    }
    fp.close();
    // 予測したカテゴリと確率(信頼度)を出力
    cv::Mat_<int>::const_iterator it = topk.begin<int>();
    for(;it!=topk.end<int>(); ++it) {
      cout << classNames[*it] << " : " << probMat.at<float>(*it) * 100 << " %" << endl;
    }
    
  } catch(const cv::Exception& e) {
    cerr << e.msg << endl;
  }
  return 0;
}
