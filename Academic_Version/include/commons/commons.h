#ifndef MDL_COMMONS_H
#define MDL_COMMONS_H

#include <map>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>

// #define NEED_DUMP true
// #define MULTI_THREAD true

/**
 * This is an empirical value indicating how many inception layers could be accelerated by multi-thread.
 */
#define MAX_INCEPTION_NUM  9
#define GPU_MODE

#ifdef ANDROID
#include <android/log.h>
#include "math/neon_mathfun.h"
#endif

// #ifndef MDL_MAC
// #include <arm_neon.h>
// #endif

#include "json/json11.h"
#include "math/math.h"
#include "commons/exception.h"

#ifdef MDL_LINUX
#include <string.h>
#include <limits.h>
#endif

#ifdef BIT_QUANTIZATION
#include "base/bit.h"
#endif


using std::min;
using std::max;
using std::map;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::stringstream;

using Json = json11::Json;

using Math = mdl::Math;

using Time = decltype(std::chrono::high_resolution_clock::now());

using MDLException = mdl::MDLException;


namespace mdl {
    extern const char *log_tag;

    extern const int string_size;

    extern const int model_version;

    extern const string matrix_name_data;

    extern const string matrix_name_test_data;

    void im2col(const float *data_im, const int channels, const int height, const int width, const int kernel_size,
                const int pad, const int stride, float *data_col);

    Time time();

    double time_diff(Time t1, Time t2);

    void idle(const char *fmt, ...);

    bool equal(float a, float b);

    void copy(int length, float* x, float* y);



};

#ifdef ANDROID
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARNING, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, mdl::log_tag, __VA_ARGS__); printf(__VA_ARGS__)
#else
#define LOGI(...) mdl::idle(__VA_ARGS__);
#define LOGW(...) mdl::idle(__VA_ARGS__);
#define LOGD(...) mdl::idle(__VA_ARGS__);
#define LOGE(...) mdl::idle(__VA_ARGS__);
#endif

/**
 * throw the c++ exception to java level
 */
#ifdef ANDROID
#define EXCEPTION_HEADER try {
#define EXCEPTION_FOOTER } catch (const MDLException &exception) {                                                   \
                            const char *message = exception.what();                                                  \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (const std::exception &exception) {                                                 \
                            const char *message = (mdl::exception_prefix + exception.what()).c_str();                \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         } catch (...) {                                                                             \
                            const char *message = (mdl::exception_prefix + "Unknown Exception.").c_str();            \
                            LOGE(message);                                                                           \
                            jclass exception_class = env->FindClass("com/baidu/mdl/demo/MDLException"); \
                            if (exception_class != NULL) {                                                           \
                                env->ThrowNew(exception_class, message);                                             \
                            }                                                                                        \
                         }
#endif

#endif
