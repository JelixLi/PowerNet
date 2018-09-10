#include "base/bit.h"

namespace mdl {

    unsigned int BitMatrix::transfer2bit(int C_row_bit_cur,float a)
    {
        int s=C_row_bit_cur%32;
        if(a>delta)
            return 2u<<s;
        else if(a<delta)
            return 3u<<s;
        else
            return 0u;
    }

    void BitMatrix::setBit(int C_row_bit_cur,int C_col_cur,float Y)
    {
        C[(C_row_bit_cur/32)*C_width+C_col_cur]^=transfer2bit(C_row_bit_cur,Y);
    }

    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }


    //
    // void bit_im2col(const float *data_im, const int channels, const int height,
    //             const int width, const int kernel_size,
    //             const int pad, const int stride, unsigned int *data_col)
    // {
    //     int output_h=(height+2*pad-kernel_size)/stride+1;
    //     int output_w=(width+2*pad-kernel_size)/stride+1;
    //     int img_size=height*width;
    //     int C_col_cur=0,C_row_bit_cur=0,C_row_bit_old=0;
    //     int loc_x,loc_y;
    //
    //     BitMatrix CBit(data_col,output_h*output_w);
    //
    //     for(int i=-pad;i<output_h;i+=stride)
    //         for(int j=-pad;j<output_w;j+=stride)
    //         {
    //             for(int x=0;x<kernel_size;++x)
    //                 for(int y=0;y<kernel_size;++y)
    //                 {
    //                     loc_x=x+i;
    //                     loc_y=y+j;
    //
    //                     C_row_bit_old=C_row_bit_cur;
    //
    //                     if(is_a_ge_zero_and_a_lt_b(loc_x,height)&&is_a_ge_zero_and_a_lt_b(loc_y,width))
    //                     {
    //                         for(int k=0;k<channels;k++)
    //                         {
    //                             CBit.setBit(C_row_bit_cur,C_col_cur,(data_im+k*img_size)[loc_x*width+loc_y]);
    //                             //C_row_bit_cur+=bit_size*kernel_col*kernel_row;
    //                             C_row_bit_cur+=18;
    //                         }
    //                     }
    //
    //                     //C_row_bit_cur=C_row_bit_old+bit_size;
    //                     C_row_bit_cur=C_row_bit_old+2;
    //
    //                 }
    //             C_col_cur++;
    //             C_row_bit_cur=0;
    //         }
    //
    // }


    //
    // void bit_im2col(const float *data_im, const int channels, const int height,
    //             const int width, const int kernel_size,
    //             const int pad, const int stride, unsigned int *data_col)
    // {
    //     int output_h=(height+2*pad-kernel_size)/stride+1;
    //     int output_w=(width+2*pad-kernel_size)/stride+1;
    //     int img_size=height*width;
    //     int C_col_cur=0,C_row_bit_cur=0,C_row_bit_old=0;
    //     int loc_x,loc_y;
    //
    //     BitMatrix CBit(data_col,output_h*output_w);
    //
    //     for(int i=-pad;i<output_h;i+=stride)
    //         for(int j=-pad;j<output_w;j+=stride)
    //         {
    //             for(int x=0;x<kernel_size;++x)
    //                 for(int y=0;y<kernel_size;++y)
    //                 {
    // 					if(x!=(kernel_size-1)||y!=(kernel_size-1)) {  //remove the first one
    //
    // 						loc_x=x+i;
    //                         loc_y=y+j;
    //
    //                         C_row_bit_old=C_row_bit_cur;
    //
    //                         if(is_a_ge_zero_and_a_lt_b(loc_x,height)&&is_a_ge_zero_and_a_lt_b(loc_y,width))
    //                         {
    //                             for(int k=0;k<channels;k++)
    //                             {
    //                                 CBit.setBit(C_row_bit_cur,C_col_cur,(data_im+k*img_size)[loc_x*width+loc_y]);
    //                                 //C_row_bit_cur+=bit_size*kernel_col*kernel_row;
    //                                 C_row_bit_cur+=9;
    //                             }
    //                         }
    //
    //                         //C_row_bit_cur=C_row_bit_old+bit_size;
    //                         C_row_bit_cur=C_row_bit_old+1;
    // 					}
    //
    //                 }
    //             C_col_cur++;
    //             C_row_bit_cur=0;
    //         }
    //
    // }
    //




    void setBit(float a,unsigned char *data_col,int cur_row_pos,int cur_col_pos) {
        if(a>0) {
            int bit_p=cur_row_pos%8;
            unsigned int mask=(1<<bit_p);
            data_col[cur_col_pos]^=mask;
        }

    }


    void bit_im2col(const float *data_im, const int channels, const int height,
                const int width, const int kernel_size,
                const int pad, const int stride, unsigned char *data_col) {

        const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
        const int output_w = (height + 2 * pad - kernel_size) / stride + 1;
        const int channel_size = height * width;
        int cur_row_pos,cur_col_pos;
        for (int channel = channels; channel--; data_im += channel_size) {
            cur_row_pos=0;
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {

                    if(kernel_row!=(kernel_size-1)||kernel_col!=(kernel_size-1)) {  //remove the first one

                        int input_row = -pad + kernel_row;
                        cur_col_pos=0;
                        for (int output_rows = output_h; output_rows; output_rows--) {
                            if (is_a_ge_zero_and_a_lt_b(input_row, height)) {
                                int input_col = -pad + kernel_col;
                                for (int output_col = output_w; output_col; output_col--) {
                                    if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                        setBit(data_im[input_row * width + input_col],data_col,cur_row_pos,cur_col_pos);
                                    }
                                    cur_col_pos++;
                                    input_col += stride;
                                }
                            }
                            input_row += stride;
                        }
                        cur_row_pos++;
                    }

                }
            }
            data_col+=output_w*output_h;
        }
    }


    BitCount::BitCount()
    {
        generate_MASK(l_MASK,r_MASK);
        generate();
    }

    void BitCount::generate_MASK(unsigned int &b,unsigned int &c)
    {
        unsigned int a=-1;
        b=a,c=a;
        unsigned int pa=1;
        unsigned int pb=2;
        for(int i=0;i<32;i+=2)
        {
            unsigned int p_a=~(pa<<i);
            unsigned int p_b=~(pb<<i);
            b=b&p_a;
            c=c&p_b;
        }
    }


    int BitCount::bit_count_init(unsigned int a)
    {
        int count;
        for(count=0;a;count++)
            a&=a-1;
        return count;
    }


    void BitCount::generate()
    {
        mpL.clear();
        mpR.clear();

        int x=1;
        int y=2;
        int l,r;
        for(int i=0;i<(1<<16);i++)
        {
            l=i&l_MASK;
            r=i&r_MASK;
            if(l)
                mpL[l]=bit_count_init(l);
            if(r)
                mpR[r]=bit_count_init(r);
        }
        mpL[0]=0;
        mpR[0]=0;
    }

    int BitCount::bit_count(unsigned int a)
    {
        unsigned int b=a&l_MASK,c=a&r_MASK;
        return mpL[b&0xFFFF]+mpL[b>>16]+mpR[c&0xFFFF]+mpR[c>>16];
    }

    void BitCount::bit_calculate(unsigned int a,int &num_1,int &num_2)
    {
        unsigned int b=a&l_MASK,c=a&r_MASK;
        num_1=mpL[b&0xFFFF]+mpL[b>>16];
        num_2=mpR[c&0xFFFF]+mpR[c>>16];
    }


    // void bit_view(unsigned int a)
    // {
    //     int st[32];
    //     int count=0;
    //     while(count<32)
    //     {
    //         st[count++]=a&1;
    //         a=(a>>1);
    //     }
    //     while(count>0)
    //         cout<<st[--count];
    //     cout<<endl;
    // }

}
