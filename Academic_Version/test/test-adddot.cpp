#include <iostream>
#include <assert.h>


using std::cin;
using std::cout;
using std::endl;

float *_shared_array_buffer;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void transformToGpuFormat(
    float *input_data,
    int input_height,
    int input_width,
    int input_channel,
    int kernel_size,
    int pad,
    int stride) {

    assert(kernel_size==3||kernel_size==4);

    float *output_data=_shared_array_buffer;

    for(int chan=0;chan<input_channel;chan++) {
         for(int row=-pad;row<input_height+pad-kernel_size+1;row+=stride) {
            for(int col=-pad;col<input_width+pad-kernel_size+1;col+=stride) {

                for(int kernel_row=0;kernel_row<kernel_size;kernel_row++) {
                    for(int kernel_col=0;kernel_col<kernel_size;kernel_col++) {

                        int new_row=row+kernel_row;
                        int new_col=col+kernel_col;

                        if(is_a_ge_zero_and_a_lt_b(new_row,input_height)&&is_a_ge_zero_and_a_lt_b(new_col,input_width)) {
                            *output_data++ = input_data[new_row*input_width+new_col];
                        } else {
                            *output_data++ = 0;
                        }               
                    }
                }

                if(kernel_size!=4) {
                    for(int i=0;i<7;i++)
                        *output_data++ = 0;
                }

            }
        }       

        input_data += input_height*input_width;
    }


}


void TransKernelToGpuFormat(
    float *kernel;
    float *new_kernel;
    int kernel_channel;
    int kernel_size;
    int kernel_num) {

    assert(kernel_size!=4);

    kernel_size = kernel_size * kernel_size;

    float *old_k = kernel;
    float *new_k = new_kernel;

    for(int num=0;num<kernel_num;num++) {
        for(int chan=0;chan<kernel_channel;chan++) {
            for(int t=0;t<kernel_size;t++) {
                *new_k++ = *old_k++;
            }

            for(int t=kernel_size;t<16;t++) {
                *new_k++ = 0;
            }
        }
    }
}

void TransToCpuFormat(
    int data_num;
    float *gpu_vec_data;
    float *cpu_data) {

    int sum=0;
    for(int i=1;i<=data_num;i++) {
        if(i%16==0) {
            *cpu_data++ = sum;
            sum = 0;
        }

        sum += *gpu_vec_data++;
    }
}



int main() {
    int input_channel=3;
    int input_height=3;
    int input_width=3;
    int kernel_size=3;
    int pad=1;
    int stride=1;

    float *input_data=new float[input_width*input_height*input_channel];

    for(int i=0;i<input_height*input_channel*input_width;i++) {
        input_data[i]=i+1;
    }

    const int output_h=(input_height+2*pad-kernel_size+1)/stride;
    const int output_w=(input_width+2*pad-kernel_size+1)/stride;
    const int channel_size=output_w*output_h;

    _shared_array_buffer = new float[output_h*output_w*16*input_channel];

    transformToGpuFormat(input_data,input_height,input_width,input_channel,kernel_size,pad,stride);

    float *p=_shared_array_buffer;

    for(int c=input_channel;c;c--) {
         for(int i=0;i<output_h;i++) {
            for(int j=0;j<output_w;j++) {
                for(int i=0;i<16;i++) {
                    cout<<*p++<<" ";
                }

                cout<<"  ";
            }
            cout<<endl;
        }       
    }

}
