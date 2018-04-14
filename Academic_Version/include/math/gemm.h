
#ifndef MDL_GEMM_H
#define MDL_GEMM_H
#include "commons/commons.h"

namespace mdl {
    constexpr const int MC = 384;

    constexpr const int KC = 384;

    constexpr const int NC = 4096;

    constexpr const int MR = 4;

    constexpr const int NR = 4;

    struct Gemmer {
        static vector<Gemmer *> gemmers;

        float A_[MC * KC] __attribute__ ((aligned (32)));

        float B_[KC * NC] __attribute__ ((aligned (32)));

        float C_[MR * NR] __attribute__ ((aligned (32)));

        float AB_[MR * NR] __attribute__ ((aligned (32)));

        void pack_MRxk(int k, const float *A, int incRowA, int incColA, float *buffer);

        void pack_A(int mc, int kc, const float *A, int incRowA, int incColA, float *buffer);

        void pack_kxNR(int k, const float *B, int incRowB, int incColB, float *buffer);

        void pack_B(int kc, int nc, const float *B, int incRowB, int incColB, float *buffer);

        void dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int incRowC, int incColC);

        void dgeaxpy(int m, int n, float alpha, const float *X, int incRowX, int incColX, float *Y, int incRowY, int incColY);

        void dgescal(int m, int n, float alpha, float *X, int incRowX, int incColX);

        void dgemm_macro_kernel(int mc, int nc, int kc, float alpha, float beta, float *C, int incRowC, int incColC);

        void dgemm_nn(int m, int n, int k, float alpha, const float *A, int incRowA, int incColA, const float *B, int incRowB, int incColB, float beta, float *C, int incRowC, int incColC);

        void sgemm(int m, int n, int k, const float *A, const float *B, float *C);

        void sgemm(int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta);
    };
};
#endif
