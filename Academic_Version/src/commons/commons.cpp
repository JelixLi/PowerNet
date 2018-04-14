

#include "commons/commons.h"

namespace mdl {
    const char *log_tag = "MDL LOG built on " __DATE__ " " __TIME__;

    const int string_size = 30;

    const int model_version = 1;

    const string matrix_name_data = "data";

    const string matrix_name_test_data = "test-data";

    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    void im2col(const float *data_im, const int channels, const int height,
                const int width, const int kernel_size,
                const int pad, const int stride, float *data_col) {
        const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
        const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
        const int channel_size = height * width;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    int input_row = -pad + kernel_row;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad + kernel_col;
                            for (int output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(data_col++) = data_im[input_row * width + input_col];
                                } else {
                                    *(data_col++) = 0;
                                }
                                input_col += stride;
                            }
                        }
                        input_row += stride;
                    }
                }
            }
        }
    }

    Time time() {
        return std::chrono::high_resolution_clock::now();
    }

    double time_diff(Time t1, Time t2) {
        typedef std::chrono::microseconds ms;
        auto diff = t2 - t1;
        ms counter = std::chrono::duration_cast<ms>(diff);
        return counter.count() / 1000.0;
    }

    void idle(const char *fmt, ...) {
    }

    bool equal(float a, float b) {
        const float EPSILON = 1e-5;
        if (fabsf(a - b) < EPSILON) {
            return true;
        }
        return false;

    }

    void copy(int length, float *x, float *y) {
        if (x != y) {
            memcpy(y, x, sizeof(float) * length);
        }
    }
};
