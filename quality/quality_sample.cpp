#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/quality.hpp"

int main(int argc, char **argv) {
    using namespace std;

    if(argc != 2) {
        cerr << "2 arguments required ./quality_sample <image path>" << endl;
        exit(1);
    }
    
    const string model_path = "./brisque_model_live.yml";
    const string range_path = "./brisque_range_live.yml";
    const cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    shared_ptr<cv::quality::QualityBase> quality(cv::quality::QualityBRISQUE::create(model_path, range_path));
    cv::Scalar score = quality->compute(input_image);
    cout << score[0] << endl;
    return 0;
}
