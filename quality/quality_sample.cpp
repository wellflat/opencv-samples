#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/quality.hpp"

int main(int argc, char **argv) {
    using namespace std;

    const string file_name = "lenna.jpg";
    const string model_path = "./brisque_model_live.yml";
    const string range_path = "./brisque_range_live.yml";
    const cv::Mat input_image = cv::imread(file_name, cv::IMREAD_COLOR);
    shared_ptr<cv::quality::QualityBase> quality(cv::quality::QualityBRISQUE::create(model_path, range_path));
    cv::Scalar result = quality->compute(input_image);
    cout << result << endl;
    return 0;
}
