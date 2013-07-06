#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <android/log.h>

#define LOG_TAG "native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

using namespace std;

extern "C" {

typedef vector<cv::KeyPoint> KeyPoints;

cv::Mat tmp, nomask, mask;
KeyPoints p_keypoints, c_keypoints;
cv::Mat p_descriptors, c_descriptors;
cv::FastFeatureDetector fast(80);
cv::StarFeatureDetector star;
cv::ORB orb(200, 1.2f, 2, 31, 0, 2, cv::ORB::FAST_SCORE, 31);
cv::FREAK freak;
cv::BFMatcher matcher(cv::NORM_HAMMING, false);
cv::Scalar red(255,0,0), green(0,255,0), blue(0,0,255),
           magenta(255, 0, 255), yellow(255, 255, 0), white(255, 255, 255);

JNIEXPORT void JNICALL Java_com_example_features_MainActivity_setup(JNIEnv*, jobject,
                                                                    jint rows, jint cols) {
  tmp.create(rows, cols, CV_8UC1);
  nomask = cv::Mat();
  mask = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1); // swap rows <-> cols
  cv::Mat roi(mask, cv::Rect(cols/2 - cols/4, rows/2 - rows/4, cols/2, rows/2));
  LOGI("rect: (%d %d)", roi.rows, roi.cols);
  roi = cv::Scalar(255);
}

JNIEXPORT void JNICALL Java_com_example_features_MainActivity_clearKeyPoints(JNIEnv*, jobject) {
  LOGI("clear keypoints");
  p_keypoints.clear();
  c_keypoints.clear();
}

/* FAST */
JNIEXPORT void JNICALL Java_com_example_features_MainActivity_detectCorners(JNIEnv*, jobject,
                                                                            jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& kps = p_keypoints;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  fast.detect(tmp, kps);
  //LOGI("keypoints num: %d", kps.size());
  for (unsigned int i=0; i<kps.size(); ++i) {
    const cv::KeyPoint& kp = kps[i];
    cv::circle(img, cv::Point(kp.pt.x, kp.pt.y), 8, red);
  }
}

/* STAR */
JNIEXPORT void JNICALL Java_com_example_features_MainActivity_star(JNIEnv*, jobject,
                                                                  jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& kps = c_keypoints;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  star.detect(tmp, kps);
  //LOGI("keypoints num: %d", kps.size());
  for (unsigned int i=0; i<kps.size(); ++i) {
    const cv::KeyPoint& kp = kps[i];
    cv::circle(img, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,255,0,255));
  }
}

/* ORB */
JNIEXPORT void JNICALL Java_com_example_features_MainActivity_trainOrb(JNIEnv*, jobject,
                                                                       jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& pkps = p_keypoints;
  cv::Mat& pdesc = p_descriptors;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  orb.detect(tmp, pkps, mask);
  orb.compute(tmp, pkps, pdesc);
}

JNIEXPORT void JNICALL Java_com_example_features_MainActivity_orb(JNIEnv*, jobject,
                                                                  jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& pkps = p_keypoints;
  KeyPoints& ckps = c_keypoints;
  cv::Mat& pdesc = p_descriptors;
  cv::Mat& cdesc = c_descriptors;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  //orb.detect(tmp, ckps);//, mask);
  //orb.compute(tmp, ckps, cdesc);
  orb(tmp, nomask, ckps, cdesc);
  vector<cv::DMatch> matches;
  matcher.match(cdesc, pdesc, matches);
  vector<cv::DMatch>::iterator base_iter(matches.begin() + matches.size()/4);
  nth_element(matches.begin(), base_iter, matches.end());
  matches.erase(base_iter + 1, matches.end());
  LOGI("previous %d current %d mached %d", pkps.size(), ckps.size(), matches.size());
  for (int i=0; i<matches.size(); ++i) {
    const cv::KeyPoint& querykp = ckps[matches[i].queryIdx];
    const cv::KeyPoint& trainkp = pkps[matches[i].trainIdx];
    cv::circle(img, cv::Point(querykp.pt.x, querykp.pt.y), 8, magenta);
    cv::circle(img, cv::Point(trainkp.pt.x, trainkp.pt.y), 2, red);
    cv::line(img,
             cv::Point(querykp.pt.x, querykp.pt.y),
             cv::Point(trainkp.pt.x, trainkp.pt.y),
             white);

  }
}

// FREAK
JNIEXPORT void JNICALL Java_com_example_features_MainActivity_trainFreak(JNIEnv*, jobject,
                                                                         jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& pkps = p_keypoints;
  cv::Mat& pdesc = p_descriptors;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  orb.detect(img, pkps, mask);
  freak.compute(img, pkps, pdesc);
  //matcher.add(pdesc);
}

JNIEXPORT void JNICALL Java_com_example_features_MainActivity_freak(JNIEnv*, jobject,
                                                                    jlong data_addr) {
  cv::Mat& img = *(cv::Mat*)data_addr;
  KeyPoints& pkps = p_keypoints;
  KeyPoints& ckps = c_keypoints;
  cv::Mat& pdesc = p_descriptors;
  cv::Mat& cdesc = c_descriptors;
  cv::cvtColor(img, tmp, CV_RGBA2GRAY);
  orb.detect(img, ckps);
  freak.compute(img, ckps, cdesc);
  vector<cv::DMatch> matches;
  matcher.match(cdesc, pdesc, matches);
  vector<cv::DMatch>::iterator base_iter(matches.begin() + matches.size()/4);
  nth_element(matches.begin(), base_iter, matches.end());
  matches.erase(base_iter + 1, matches.end());
  LOGI("previous %d current %d matched %d", pkps.size(), ckps.size(), matches.size());
  for (int i=0; i<matches.size(); ++i) {
    const cv::KeyPoint& querykp = ckps[matches[i].queryIdx];
    const cv::KeyPoint& trainkp = pkps[matches[i].trainIdx];
    cv::circle(img, cv::Point(querykp.pt.x, querykp.pt.y), 8, yellow);
    cv::circle(img, cv::Point(trainkp.pt.x, trainkp.pt.y), 2, red);
    cv::line(img,
             cv::Point(querykp.pt.x, querykp.pt.y),
             cv::Point(trainkp.pt.x, trainkp.pt.y),
             white);
   }
}
}
