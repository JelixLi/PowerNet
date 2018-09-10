__kernel void gemm_float(
    const int Mdim,
    const int Ndim,
    const int Kdim,
    __global float *A,
    __global float *B,
    __global float *C) {

        int k,j;
        int i=get_global_id(0);

        float N=Ndim;
        float K=Kdim;

        int it;
        int jt;

        float tmp;
        float Awrk[500];
        if(i<Mdim) {
            for(k=0;k<Kdim;k++) {
                it=i*K+k;
                Awrk[k]=A[it];
            }
            for(j=0;j<Ndim;j++) {
                tmp=0.0;
                for(k=0;k<Kdim;k++) {
                    jt=k*N+j;
                    tmp+=Awrk[k]*B[jt];
                }
                it=i*N+j;
                C[it]=tmp;
            }
        }

    }
