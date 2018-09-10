#include <stdio.h>
#include <iostream>

using namespace std;

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


void Debug_print(float *A,int n,int m)
{
    for(int i=0;i<n*m;i++)
     {
         cout<<A[i]<<" ";
         if((i+1)%m==0)
           cout<<endl;
     }
     cout<<endl;
}

void My_im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_size,
            const int pad, const int stride, float *data_col)
{
    const int channel_size = height * width;
    int guard=channel_size+1;
    int loc_x;
    int loc_y;
    for (int channel=channels;channel--;data_im+=channel_size)
    {
        for(int row=0;row<height;row++)
        {
            loc_x=row-pad;
            if(loc_x<0)
                loc_x=height;
            for(int col=0;col<width;col++)
            {
                loc_y=col-pad;
                if(loc_y<0)
                   loc_y=0;

            }
        }
    }
}

int main()
{
    // const int n=4;
    // float A[n*n];
    // float B[100];
    // for(int i=0;i<n*n;i++)
    //  {
    //      A[i]=i+1;
    //  }
    //  Debug_print(A,n,n);
    //  My_im2col(A,1,n,n,3,0,1,B);
    //  Debug_print(B,4,9);

    int k=3;
    for(int i=0;i<k;i++)
      for(int j=0;j<k;j++)
      {
          if(i==0&&j!=0)
          {
                printf("*(data_col++)=Contain(locx,locy+%d);\n",j);
          }
          if(i!=0&&j==0)
          {
                printf("*(data_col++)=Contain(locx+%d,locy);\n",i);
          }
          if(i==0&&j==0)
          {
                printf("*(data_col++)=Contain(locx,locy);\n");
          }
          if(i!=0&&j!=0)
          {
            printf("*(data_col++)=Contain(locx+%d,locy+%d);\n",i,j);
          }
      }
}

/*
*(data_col++)=Contain(locx,locy);
*(data_col++)=Contain(locx,locy+1);
*(data_col++)=Contain(locx,locy+2);
*(data_col++)=Contain(locx+1,locy);
*(data_col++)=Contain(locx+1,locy+1);
*(data_col++)=Contain(locx+1,locy+2);
*(data_col++)=Contain(locx+2,locy);
*(data_col++)=Contain(locx+2,locy+1);
*(data_col++)=Contain(locx+2,locy+2);
*/
