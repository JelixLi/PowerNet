
#ifndef MDL_BIT_GEMM_H
#define MDL_BIT_GEMM_H
#include "commons/commons.h"

namespace mdl {

    void generate();

    void dgemm_nn(int m,int n,int k,const unsigned int *A,int incRowA,int incColA,const unsigned int *B,int incRowB,int incColB,
             float *C,int incRowC,int incColC);

};
#endif
