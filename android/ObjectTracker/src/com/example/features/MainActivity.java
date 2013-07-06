package com.example.features;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

public class MainActivity extends Activity implements CvCameraViewListener {
  private final String TAG = "MainActivity";
  private final String MODULE = "ObjectTracker";
  private CameraBridgeViewBase mCameraView; 
  private int mViewMode;
  private int mIsTrain;
  
  private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
    @Override
    public void onManagerConnected(int status) {
      switch (status) {
        case LoaderCallbackInterface.SUCCESS:
          Log.i(TAG, "OpenCV loaded successfully");
          System.loadLibrary(MODULE);
          mCameraView.enableView();
          break;
        default:
          super.onManagerConnected(status);
          break;
        }
      }
  };
  
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    
    mCameraView = (CameraBridgeViewBase)findViewById(R.id.camera_view);
    mCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
    mViewMode = 0;
    mIsTrain = 0;
    mCameraView.setCvCameraViewListener(this);
    
    if (BuildConfig.DEBUG) {
      Toast.makeText(this, "Debug Mode", Toast.LENGTH_LONG).show();
    } else {
      Toast.makeText(this, "Release Mode", Toast.LENGTH_LONG).show();
    }
  }
  
  @Override
  public void onPause() {
    if (mCameraView != null) {
      mCameraView.disableView();
    }
    super.onPause();
  }

  @Override
  public void onResume() {
    super.onResume();
    OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
  }
  
  @Override
  public void onDestroy() {
    super.onDestroy();
    if (mCameraView != null) {
      mCameraView.disableView();
    }
  }
  

  @Override
  public boolean onCreateOptionsMenu(Menu menu) {
    menu.add(Menu.NONE, 0, 0, "RGBA");
    menu.add(Menu.NONE, 1, 1, "FAST");
    menu.add(Menu.NONE, 2, 2, "ORB");
    menu.add(Menu.NONE, 3, 3, "FREAK");
    //getMenuInflater().inflate(R.menu.activity_main, menu);
    return true;
  }
  
  @Override
  public boolean onOptionsItemSelected(MenuItem item) {
    switch (item.getItemId()) {
    case 0:
      mCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
      mViewMode = 0;
      break;
    case 1:
      mCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
      mViewMode = 1;
      break;
    case 2:
      mCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
      mViewMode = 2;
      break;
    case 3:
      mCameraView.SetCaptureFormat(Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
      mViewMode = 3;
      break;
    }
    mIsTrain = 0;
    return true;
  }

  @Override
  public void onCameraViewStarted(int width, int height) {
    setup(height, width);
  }

  @Override
  public void onCameraViewStopped() {

  }

  @Override
  public Mat onCameraFrame(Mat inputFrame) {
    switch (mViewMode) {
    case 0:
      int h = inputFrame.rows(), w = inputFrame.cols();
      Core.rectangle(inputFrame, new Point(w/2-w/4,h/2-h/4), new Point(w/2+w/4,h/2+h/4), new Scalar(255));
      return inputFrame;
    case 1:
      detectCorners(inputFrame.getNativeObjAddr());
      return inputFrame;
    case 2:
      if (mIsTrain == 1) {
        orb(inputFrame.getNativeObjAddr());
      } else {
        trainOrb(inputFrame.getNativeObjAddr());
        mIsTrain = 1;
      }
      return inputFrame;
    case 3:
      if (mIsTrain == 1) {
        freak(inputFrame.getNativeObjAddr());
      } else {
        trainFreak(inputFrame.getNativeObjAddr());
        mIsTrain = 1;
      }
      return inputFrame;
    default:
      return inputFrame;
    }
  }
  
  public native void setup(int rows, int cols);
  public native void detectCorners(long dataAddr);
  public native void star(long dataAddr);
  public native void orb(long dataAddr);
  public native void trainOrb(long dataAddr);
  public native void freak(long dataAddr);
  public native void trainFreak(long dataAddr);
}
