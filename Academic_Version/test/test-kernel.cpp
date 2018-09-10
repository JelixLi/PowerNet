#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <string.h>

using namespace std;

char mpH[65550];

int pre_count(int a) {
	int cnt=0;
	for(int b=a;b;b=(b-1)&b) {
		cnt++;
	}
	return cnt;
}

void generate() {
	for(int i=0;i<(1<<8);i++) {
		mpH[i]=pre_count(i);
	}
}

int bit_count(unsigned short int a) {
	return mpH[a];
}


#define MC 384
#define KC 384
#define NC 4096

#define MR 4
#define NR 4

unsigned short int _A[MC*KC];
unsigned short int _B[KC*NC];
float _C[MR*NR];

static void pack_MRxk(int k,const unsigned short int *A,int incRowA,int incColA,
    unsigned short int *buffer)
{
    // int i,j;
    // for(j=0;j<k;j++)
    // {
    //     for(i=0;i<MR;i++)
    //     {
    //         buffer[i]=A[i*incRowA];
    //     }
    //     buffer+=MR;
    //     A+=incColA;
    // }

	int j, a2 = incRowA, a3 = 2 * incRowA, a4 = 3 * incRowA;
	for (j = 0; j < k; ++j) {
		buffer[0] = A[0];
		buffer[1] = A[a2];
		buffer[2] = A[a3];
		buffer[3] = A[a4];
		A += 1;
		buffer += MR;
	}
}


static void pack_A(int mc, int kc, const unsigned short int *A, int incRowA, int incColA,
       unsigned short int *buffer)
{
    int mp=mc/MR;
    int _mr=mc%MR;
	int tmp1 = kc * MR;
	int tmp2 = MR * incRowA;

    int i,j;

    for(i=0;i<mp;++i)
    {
        pack_MRxk(kc,A,incRowA,incColA,buffer);
        // buffer+=kc*MR;
        // A+=MR*incRowA;
		buffer+=tmp1;
		A+=tmp2;
    }
    if (_mr>0) {
        for(j=0;j<kc;++j)
        {
            for(i=0;i<_mr;++i)
            {
                buffer[i]=A[i*incRowA];
            }
            for(i=_mr;i<MR;++i)
            {
                buffer[i]=0.0;
            }
            buffer+=MR;
            A+=incColA;
        }
    }
}


static void pack_kxNR(int k, const unsigned short int *B, int incRowB, int incColB,
          unsigned short int *buffer)
{
    int i,j;

    for(i=0;i<k;++i)
    {
        for (j=0;j<NR;++j)
            buffer[j] = B[j*incColB];

        buffer+=NR;
        B+=incRowB;
    }
}

static void pack_B(int kc,int nc,const unsigned short int *B,int incRowB,int incColB,
    unsigned short int *buffer)
{
    int np=nc/NR;
    int _nr=nc%NR;
	int tmp1=kc*NR;
    int i,j;

    for(j=0;j<np;j++)
    {
        pack_kxNR(kc,B,incRowB,incColB,buffer);
        // buffer+=kc*NR;
		buffer+=tmp1;
        B+=incColB*NR;
    }

    if(_nr>0)
    {
        for(i=0;i<kc;i++)
        {
            for(j=0;j<_nr;j++)
                buffer[j]=B[incColB*j];
            for(j=_nr;j<NR;j++)
                buffer[j]=0.0;
            buffer+=NR;
            B+=incRowB;
        }
    }
}

static void dgemm_micro_kernel(int kc,const unsigned short int *A, const unsigned short int *B, float *C,
     int incRowC, int incColC)
{
    int AB[MR*NR];

    int i,j,l;

    for(l=0;l<MR*NR;++l)
        AB[l] = 0;

    for (l=0;l<kc;++l)
    {
        for(j=0;j<NR;++j)
        {
            for(i=0;i<MR;++i)
                AB[i+j*MR]+=mpH[A[i]&B[j]];
        }
        A+=MR;
        B+=NR;
    }

    for(j=0;j<NR;++j)
    {
        for(i=0;i<MR;++i)
        {
            C[i*incRowC+j*incColC]+=AB[i+j*MR];
        }
    }
}

static void dgeaxpy(int m,int n,const float *X,int incRowX,int incColX,float *Y,int incRowY,int incColY)
{
    int i,j;

    for(j=0;j<n;j++)
        for(i=0;i<m;i++)
            Y[i*incRowY+j*incColY]+=X[i*incRowX+j*incColX];
}

static void dgemm_macro_kernel(int mc,int nc,int kc,float *C,
    int incRowC,int incColC)
{
    int mp=(mc+MR-1)/MR;
    int np=(nc+NR-1)/NR;

    int _mr=mc%MR;
    int _nr=nc%NR;

    int mr,nr;
    int i,j;

    for(j=0;j<np;j++)
    {
        nr=(j!=np-1||_nr==0)?NR:_nr;
        for(i=0;i<mp;i++)
        {
            mr=(i!=mp-1||_mr==0)?MR:_mr;
            if(mr==MR&&nr==NR)
                dgemm_micro_kernel(kc,&_A[i*kc*MR],&_B[j*kc*NR],&C[i*MR*incRowC+j*NR*incColC],incRowC,incColC);
            else
            {
                dgemm_micro_kernel(kc,&_A[i*kc*MR],&_B[j*kc*NR],_C,1,MR);
                dgeaxpy(mr,nr,_C,1,MR,&C[i*MR*incRowC+j*NR*incColC],incRowC,incColC);
            }
        }
    }
}



void dgemm_nn(int m,int n,int k,const unsigned short int *A,int incRowA,int incColA,const unsigned short int *B,int incRowB,int incColB,
         float *C,int incRowC,int incColC)
{
    int mb=(m+MC-1)/MC;
    int nb=(n+NC-1)/NC;
    int kb=(k+KC-1)/KC;

    int _mc=m%MC;
    int _nc=n%NC;
    int _kc=k%KC;

    int mc,nc,kc;
    int i,j,l;

    for (j=0;j<nb;++j)
    {
        nc=(j!=nb-1||_nc==0)?NC:_nc;

        for(l=0;l<kb;++l)
        {
            kc=(l!=kb-1||_kc==0)?KC:_kc;

            pack_B(kc,nc,&B[l*KC*incRowB+j*NC*incColB],incRowB,incColB,_B);

            for (i=0;i<mb;++i) {
                mc=(i!=mb-1||_mc==0)?MC:_mc;

                pack_A(mc,kc,&A[i*MC*incRowA+l*KC*incColA],incRowA,incColA,_A);

                dgemm_macro_kernel(mc,nc,kc,&C[i*MC*incRowC+j*NC*incColC],incRowC,incColC);
            }
        }
    }
}


void Init(unsigned short int *A,int m,int n)
{
    //srand(0);
    int i;
    for(i=0;i<m*n;i++)
    {
        //A[i]=unsigned(rand()%100);
        A[i]=i+1;
    }
}

void gemm_norm(const unsigned short int *A,const unsigned short int *B,float *C,int m,int n,int k)
{
    int i,j,s;

    for(i=0;i<m;i++)
      for(j=0;j<n;j++)
      {
          for(s=0;s<k;s++)
          {
              C[i*n+j]+=A[i*k+s]*B[s*n+j];
          }
      }
}

void check(const float *A,const float *B,int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        if(int(A[i])!=int(B[i]))
        {
            cout<<"error"<<endl;
            return;
        }
    }
    cout<<"success"<<endl;
}



const int m=1024;
const int n=1024;
const int k=64;

unsigned short int A[m*k];
unsigned short int B[k*n];
float C[m*n];
float C_ST[m*n];


int main()
{
    Init(A,m,k);
    Init(B,k,n);
    for(int i=0;i<m*n;i++)
    {
        C[i]=C_ST[i]=0;
    }

    // unsigned short int Buffer_A[m*n];
    // unsigned short int Buffer_B[m*n];
    // pack_MRxk(k,A,k,1,Buffer_A);
    // pack_kxNR(k,B,n,1,Buffer_B);
    // dgemm_micro_kernel(k,Buffer_A,Buffer_B,C,n,1);
	//
	generate();
	for(int term=0;term<10;term++) {
		clock_t start=clock();
	    dgemm_nn(m,n,k,A,k,1,B,n,1,C,n,1);
	    clock_t end=clock();
	    cout<<double(end-start)/CLOCKS_PER_SEC<<endl;
	}

    //gemm_norm(A,B,C_ST,m,n,k);
    //check(C,C_ST,m*n);

    // debug_print(A,m,k);
    // // debug_print(Buffer_A,m,n);
    // debug_print(B,k,n);
    // // debug_print(Buffer_B,m,n);
    // debug_print(C,m,n);
    //
    // debug_print(C_ST,m,n);
}