#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char **argv) {
    using namespace std;

    const string file_name = "qrcode_sample.png";
    const cv::Mat input_image = cv::imread(file_name, cv::IMREAD_COLOR);
    cv::Mat output_image = input_image.clone();
    vector<cv::Point> points;
    cv::Mat straight_qrcode;
    // QRコード検出器
    cv::QRCodeDetector detector;
    // QRコードの検出と復号化(デコード)
    const string data = detector.detectAndDecode(input_image, points, straight_qrcode);
    if(data.length() > 0) {
        // 復号化情報(文字列)の出力
        cout << "decoded data: " << data << endl;
        // 検出結果の矩形描画
        for(size_t i = 0; i < points.size(); ++i) {
            cv::line(output_image, points[i], points[(i + 1) % points.size()], cv::Scalar(0, 0, 255), 4);
        }
        cv::imwrite("output.png", output_image);
        // おまけでQRコードのバージョンも計算
        cout << "QR code version: " << ((straight_qrcode.rows - 21) / 4) + 1 << endl;
    } else {
        cout << "QR code not detected" << endl;
    }
    return 0;
}
