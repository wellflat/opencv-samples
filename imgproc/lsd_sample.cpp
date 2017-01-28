#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    std::string fileName = "sample.jpg";
    Mat image = imread(fileName, IMREAD_GRAYSCALE);
    if(image.empty()) {
        return -1;
    }
#if 1
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);
#else
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_NONE);
#endif

    double start = double(getTickCount());
    vector<Vec4f> lines_std;

    lsd->detect(image, lines_std);

    double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
    std::cout << "It took " << duration_ms << " ms." << std::endl;

    Mat output(image);
    lsd->drawSegments(output, lines_std);

    imwrite("output.jpg", output);

    return 0;
}
