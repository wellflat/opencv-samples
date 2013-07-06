LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include ../../../Library/OpenCV-2.4.3.2-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := ObjectTracker
LOCAL_SRC_FILES := jni_part.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
