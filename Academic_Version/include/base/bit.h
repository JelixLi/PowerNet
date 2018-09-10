#ifndef BIT_H
#define BIT_H


#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <map>

using std::map;

namespace mdl {

    const float delta=1.0;

    class BitMatrix
    {
    public:
        BitMatrix(unsigned int *C,int incRowC)
        {
            this->C=C;
            C_width=incRowC;
        }
        void setBit(int C_row_bit_cur,int C_col_cur,float Y);
        unsigned int transfer2bit(int C_row_bit_cur,float a);

    private:
        unsigned int *C;
        int C_width;
    };


    void bit_im2col(const float *data_im, const int channels, const int height,
                const int width, const int kernel_size,
                const int pad, const int stride, unsigned char *data_col);


    class BitCount
    {
    public:
        BitCount();

        void generate_MASK(unsigned int &b,unsigned int &c);
        int bit_count_init(unsigned int a);
        void generate();

        int bit_count(unsigned int a);
        void bit_calculate(unsigned int a,int &num_1,int &num_2);
    private:
        unsigned int l_MASK;
        unsigned int r_MASK;
        map<int,char> mpL;
        map<int,char> mpR;
    };


}

#endif
