#ifndef Power_COMMONS_H
#define Power_COMMONS_H

#include <map>
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>


#include "json/json11.h"
#include "math/math.h"
#include "commons/exception.h"
#include <string.h>
#include <limits.h>


#ifdef GPU
#include "QPULib.h"
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

using Math = Power::Math;

using Time = decltype(std::chrono::high_resolution_clock::now());

using PowerException = Power::PowerException;


namespace Power {
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


#endif
