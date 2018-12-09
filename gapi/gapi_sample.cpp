#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char **argv) {
    using namespace std;

    const string file_name = "lenna.jpg";
    const cv::Mat input_image = cv::imread(file_name, cv::IMREAD_COLOR);
    if(!input_image.empty()) {
        cv::Mat output_image = input_image.clone();
        const cv::GMat in;
        const cv::GMat vga = cv::gapi::resize(in, cv::Size(), 0.5, 0.5);
        const cv::GMat gray = cv::gapi::BGR2Gray(vga);
        const cv::GMat blurred = cv::gapi::blur(gray, cv::Size(5,5));
        const cv::GMat edges = cv::gapi::Canny(blurred, 32, 128, 3);
        cv::GMat b,g,r;
        tie(b,g,r) = cv::gapi::split3(vga);
        const cv::GMat out = cv::gapi::merge3(b, g | edges, r);
        cv::GComputation ac(in, out);
        ac.apply(input_image, output_image);
        cv::imwrite("gapi_output.jpg", output_image);
    } else {
        cout << "image can't read" << endl;
    }
    return 0;
}
