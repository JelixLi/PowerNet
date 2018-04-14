
#ifdef ANDROID

#include <jni.h>

#ifndef MOBILE_DEEP_LEARNING_CAFFE_JNI_H
#define MOBILE_DEEP_LEARNING_CAFFE_JNI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * load model & params of the net for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_mdl_demo_MDL_load(
        JNIEnv *env, jclass thiz, jstring modelPath, jstring weightsPath);

/**
 * object detection for anroid
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_mdl_demo_MDL_predictImage(
        JNIEnv *env, jclass thiz, jfloatArray buf);

/**
 * set thread num
 */
JNIEXPORT void JNICALL Java_com_baidu_mdl_demo_MDL_setThreadNum(
        JNIEnv *env, jclass thiz, jint num);

/**
 * clear data of the net when destroy for android
 */
JNIEXPORT void JNICALL Java_com_baidu_mdl_demo_MDL_clear(
        JNIEnv *env, jclass thiz);
/**
 * validate wheather the device is fast enough for obj detection for android
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_mdl_demo_MDL_validate(
        JNIEnv *env, jclass thiz);

#ifdef __cplusplus
}
#endif

#endif //MOBILE_DEEP_LEARNING_CAFFE_JNI_H

#endif
