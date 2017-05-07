#include <iostream>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "2 arugments required" << endl;
        return -1;
    }
    char* fileName = argv[1];
    Mat image = imread(fileName, IMREAD_GRAYSCALE);
    if(image.empty()) {
        return -1;
    }
    int type = LSD_REFINE_STD; //LSD_REFINE_STD,LSD_REFINE_NONE
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(type, 0.8);
    cout << image.size() << endl;
    Mat angles = Mat_<double>(image.size());
    cout << angles.cols << ":" << angles.rows << endl;
    Mat row = angles.row(0);
    cout << row.cols << ":" << row.rows << endl;

    
    
    vector<Vec4f> lines;
    bool is_lsd = true;
    Mat src, dst, color_dst;
    
    if(is_lsd) {
        double start = double(getTickCount());
        lsd->detect(image, lines);
        double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "It took " << duration_ms << " ms." << std::endl;
        cout << "lines: " << lines.size() << endl;
        Mat output(image);
        lsd->drawSegments(output, lines);
        //imwrite("tmp/output.jpg", output);
    } else {
        double start = double(getTickCount());
        Canny(image, dst, 100, 200, 3);
        cvtColor(image, color_dst, COLOR_GRAY2BGR);
        HoughLinesP(dst, lines, 1, CV_PI/180, 100, 80, 10);
        double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
        std::cout << "It took " << duration_ms << " ms." << std::endl;
        for(size_t i = 0; i < lines.size(); i++) {
            line(color_dst, Point(lines[i][0], lines[i][1]),
                 Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1.5, 8);
        }
        //imwrite("tmp/hough.jpg", color_dst);
    }
    return 0;
}
