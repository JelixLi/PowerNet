
#ifndef Power_MATH_H
#define Power_MATH_H

#include "commons/commons.h"
#include <array>
#include <cmath>

namespace Power {


    struct Math {
        inline static void v_sqr(const int n, const float *xs, float *ys) {
            for (int i = 0; i < n; i++) {
                ys[i] = xs[i] * xs[i];
            }
        }

        inline static void axpy(const int n, const float alpha, const float *xs, float *ys) {
            for (int i = 0; i < n; i++) {
                ys[i] = alpha * xs[i] + ys[i];
            }
        }

        inline static void v_pow(const int n, const float *xs, const float p, float *ys) {
            for (int i = 0; i < n; i++) {
                ys[i] = pow(xs[i], p);
            }
        }

        inline static void v_mul(const int n, const float *xs, const float *ys, float *zs) {
            for (int i = 0; i < n; i++) {
                zs[i] = xs[i] * ys[i];
            }
        }

        // static void transpose(const int m, const int n, float *xs) {
        //     float *trans_data = new float[m * n];
        //     for (int i = 0; i < m; i++) {
        //         for (int j = 0; j < n; j++) {
        //             trans_data[j * m + i] = xs[i * n + j];
        //         }
        //     }
        //     for (int i = 0; i < n; i++) {
        //         for (int j = 0; j < m; j++) {
        //             xs[i * m + j] = trans_data[i * m + j];
        //         }
        //     }
        //     delete[] trans_data;
        // }

        inline static void v_scale(const int n, const float scale_factor, const float *data, float *dest) {
            for (int i = 0; i < n; ++i) {
                dest[i] = data[i] * scale_factor;

            }

        }

        inline static void v_add(const int N, const float alpha, float *Y) {
            for (int i = 0; i < N; ++i) {
                Y[i] += alpha;

            }
        }

        inline static void v_div(const int n, const float *a, const float *b, float *y) {
            for (int i = 0; i < n; ++i) {
                if (b[i] != 0) {
                    y[i] = a[i] / b[i];
                }
            }
        }

        inline static void v_exp(const int n, const float *a, float *y) {

            for (int i = 0; i < n; ++i) {
                    y[i] = expf(a[i]);

            }


        }
    };
};

#endif
