
#include "layer/convolution_layer.h"
#include "math/gemm.h"
#include <fstream>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include "base/bit.h"
#include "math/bit_gemm.h"
#include <assert.h>

namespace mdl {

#ifdef GPU_MODE

    void TransKernelToGpuFormat(
        float *kernel,
        float *new_kernel,
        int kernel_channel,
        int kernel_size,
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

#endif

    ConvolutionLayer::ConvolutionLayer(const Json &config) : Layer(config), _col_buffer(nullptr),
                                                             _bias_buffer(nullptr) {
        assure_memory();
        _layer_type = LayerType::CONVOLUTION;
        _pid = config["pid"].int_value();
        auto &param = config["param"];
        _output_num = param["output_num"].int_value();
        _kernel_size = param["kernel_size"].int_value();
        _pad = param["pad"].int_value();
        _stride = param["stride"].int_value();
        // _bias_term default = 1
        if (param.object_items().count("bias_term")) {
            _bias_term = param["bias_term"].int_value();
        } else {
            _bias_term = 1;
        }
        // _group default = 1
        if (param.object_items().count("group")) {
            _group = param["group"].int_value();
        } else {
            _group = 1;
        }
        _need_im2col = !((_kernel_size == 1) && (_pad == 0) && (_stride == 1));
        if (_need_im2col) {
            _col_buffer = new Matrix();
            _col_buffer->resize({_weight[0]->count(1) * _group, _output[0]->dimension(2), _output[0]->dimension(3)});
            _col_buffer->reallocate();
        }
        _bias_buffer = new Matrix();
        _bias_buffer->resize({_output[0]->count(2)});
        _bias_buffer->reallocate();
        std::fill(_bias_buffer->get_data(), _bias_buffer->get_data() + _bias_buffer->count(), 1.0);

#ifdef XNOR_MODE

        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);
        _sum_buffer=new float[input_height*input_width];
        int owh = _output[0]->count(2);  // output  width * height
        _mlp_buffer=new float[owh];
        int kernel_h = _weight[0]->dimension(2);
        int kernel_w = _weight[0]->dimension(3);
        _mlp_kernel_buffer=new float[kernel_h*kernel_w];
        for(int i=0;i<kernel_h*kernel_w;i++) {
            (*_mlp_kernel_buffer++)=1/(input_height*input_width);
        }
        _sum_col_data=new float[owh*kernel_h*kernel_w];


#endif

#ifdef GPU_MODE

       int kernel_channel = _weight[0]->dimension(1);
       int kernel_size = _weight[0]->dimension(2);
       int kernel_num = _weight[0]->count(0);

       _gpu_kernel_buffer = new float[kernel_num*16];
       TransKernelToGpuFormat(_weight[0]->get_data(),_gpu_kernel_buffer,kernel_channel,kernel_size,_output_num);

       _gpu_input_data_buffer = new float[_input[0]->count(1)*16];
       _gpu_output_data_buffer = new float[_output[0]->count(1)*16];


#endif


    }

#ifdef GPU_MODE

    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    void transformToGpuFormat(
        float *_shared_array_buffer,
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

    void TransToCpuFormat(
        int data_num,
        float *gpu_vec_data,
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


    void ConvolutionLayer::forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num) {

        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);


        transformToGpuFormat(_gpu_input_data_buffer,_input[0]->get_data(),input_height,input_width,input_channel,_kernel_size,_pad,_stride);


        for (int i = 0; i < _group; ++i) {
            
            //gpu_gemm()

        }

        TransToCpuFormat(_output[0]->count(1),_gpu_output_data_buffer,_output[0]->get_data());

    }

#endif




    ConvolutionLayer::~ConvolutionLayer() {
        if (_col_buffer != nullptr) {
            delete _col_buffer;
            _col_buffer = nullptr;
        }
        if (_bias_buffer != nullptr) {
            delete _bias_buffer;
            _bias_buffer = nullptr;
        }
    }

    void run1(int id, int m, int n, int k, const float *A, const float *B, float *C) {
        Gemmer::gemmers[id]->sgemm(m, n, k, A, B, C);
    }

    void run2(int id, int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta) {
        Gemmer::gemmers[id]->sgemm(m, n, k, A, B, C, alpha, beta);
    }

    void ConvolutionLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *weight_data = _weight[0]->get_data();

        int input_offset = _input[0]->count(1);
        int output_offset = _output[0]->count(1);

        int input_num = _input[0]->count(0, 1);

        for (int i = 0; i < input_num; i++) {
            forward_gemm(input_data + i * input_offset, weight_data, output_data + i * output_offset, thread_num);
            if (_bias_term == 1) {
                float *bias_data = _weight[1]->get_data();
                forward_bias(output_data + i * output_offset, bias_data, thread_num);
            }

        }
        descript();
    }





#ifdef XNOR_MODE
    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    void ConvolutionLayer::Compute_Mlp(int input_height,int input_width) {
        int n = _output[0]->count(2);  // output  width * height
        int k = _weight[0]->count(2);   //  kernel_h * kernel_w

        if(_need_im2col) {
            im2col(_sum_buffer,1, input_height, input_width, _kernel_size, _pad, _stride, _sum_col_data);
        }


        int _a=n/4;
        int _b=n%4;


        register float *ptr_1=_sum_col_data;
        register float *ptr_mlp=_mlp_buffer;
        int data_size=input_height*input_width;
        float t=1/data_size;

        int i,j;

        for(i=0;i<k;i++) {
            ptr_mlp=_mlp_buffer;

            for(int inc=0;inc<_a;inc++) {
                *ptr_mlp+=*ptr_1;
                *(ptr_mlp+1)+=*(ptr_1+1);
                *(ptr_mlp+2)+=*(ptr_1+2);
                *(ptr_mlp+3)+=*(ptr_1+3);
            }

            for(int inc=0;inc<_b;inc++) {
                *ptr_mlp+=*ptr_1;
            }

        }

        ptr_mlp=_mlp_buffer;

        for(int inc=0;inc<_a;inc++) {
            *(ptr_mlp)=*(ptr_mlp)*t;
            *(ptr_mlp+1)=*(ptr_mlp+1)*t;
            *(ptr_mlp+2)=*(ptr_mlp+2)*t;
            *(ptr_mlp+3)=*(ptr_mlp+3)*t;
        }

        for(int inc=0;inc<_b;inc++) {
            *(ptr_mlp)=*(ptr_mlp)*t;
        }


        // if(_need_im2col) {
        //     im2col(_sum_buffer,1, input_height, input_width, _kernel_size, _pad, _stride, _sum_col_data);
        // }
        //cblas_sgemv(CblasRowMajor,CblasTrans,n,k,1,_sum_col_data,k,_mlp_kernel_buffer,1,0,_mlp_buffer,1);
        // Gemmer::gemmers[0]->sgemm(1,n,k,_mlp_kernel_buffer,
        //                                     _sum_col_data,_mlp_buffer);

        // int data_size=input_height*input_width;
        // register float *p=_sum_buffer,*q=_sum_buffer+data_size,*r=_sum_buffer+data_size*2;
        // register float *m=_mlp_buffer;
        // register float sum_1,sum_2,sum_3;
        // register float t=1/data_size;
        // for(int i=-_pad;i<input_height+_pad-_stride+1;i+=_stride) {
        //     for(int j=-_pad;j<input_width+_pad-_stride+1;j+=_stride) {
        //
        //         if(is_a_ge_zero_and_a_lt_b(i,input_height)&&is_a_ge_zero_and_a_lt_b(j,input_width)) {
        //             // for(int k=0;k<_kernel_size;k++) {
        //             //     sum_1+=*p++;
        //             //     sum_2+=*q++;
        //             //     sum_3+=*r++;
        //             // }
        //             sum_1=*p+*(p+1)+*(p+2);
        //             sum_2=*q+*(q+1)+*(q+2);
        //             sum_3=*r+*(r+1)+*(r+2);
        //             p++; q++; r++;
        //             (*m++)=(sum_1+sum_2+sum_3)*t;
        //         }
        //
        //     }
        // }



    }

    void  ConvolutionLayer::Compute_Sum(float *input_data,int input_height,int input_width,int input_channel) {
        float *p=input_data;
        register float *sum;
        int data_size=input_width*input_height;
        int _n=data_size/4;
        int _m=data_size%4;

        int _k=input_channel/4;
        int _r=input_channel%4;


        // for(int i=0;i<input_channel;i++) {
        //     sum=_sum_buffer;
        //     for(int j=0;j<_n;j++) {
        //         *sum+=*p;
        //         *(sum+1)+=*(p+1);
        //         *(sum+2)+=*(p+2);
        //         *(sum+3)+=*(p+3);
        //
        //         sum+=4;
        //         p+=4;
        //     }
        //     for(int k=0;k<_m;k++) {
        //         (*sum++)+=*p++;
        //     }
        // }


        register float *ptr_1;
        register float *ptr_2;
        register float *ptr_3;
        register float *ptr_4;

        for(int inc=0;inc<_k;inc++) {
            sum=_sum_buffer;

            ptr_1=p+inc*4*data_size;
            ptr_2=p+(inc*4+1)*data_size;
            ptr_3=p+(inc*4+2)*data_size;
            ptr_4=p+(inc*4+3)*data_size;

            for(int j=0;j<_n;j++) {
                *sum+=*ptr_1+*ptr_2+*ptr_3+*ptr_4;
                *(sum+1)+=*(ptr_1+1)+*(ptr_2+1)+*(ptr_3+1)+*(ptr_4+1);
                *(sum+2)+=*(ptr_1+2)+*(ptr_2+2)+*(ptr_3+2)+*(ptr_4+2);
                *(sum+3)+=*(ptr_1+3)+*(ptr_2+3)+*(ptr_3+3)+*(ptr_4+3);

                sum+=4;
                ptr_1+=4;
                ptr_2+=4;
                ptr_3+=4;
                ptr_4+=4;
            }

            for(int k=0;k<_m;k++) {
                (*sum++)+=*ptr_1+*ptr_2+*ptr_3+*ptr_4;
            }
        }



        for(int inc=0;inc<_r;inc++) {
            sum=_sum_buffer;

            ptr_1=p+inc*data_size;

            for(int j=0;j<_n;j++) {
                *sum+=*ptr_1;
                *(sum+1)+=*(ptr_1+1);
                *(sum+2)+=*(ptr_1+2);
                *(sum+3)+=*(ptr_1+3);

                sum+=4;
                ptr_1+=4;
            }

            for(int k=0;k<_m;k++) {
                (*sum++)+=*(ptr_1++);
            }
        }



        int n = _output[0]->count(2);  // output  width * height
        int k = _weight[0]->count(2);   //  kernel_h * kernel_w


        register float *ptr_mlp=_mlp_buffer;
        data_size=input_height*input_width;
        register float t=1/data_size;

        if(_need_im2col) {
            im2col(_sum_buffer,1, input_height, input_width, _kernel_size, _pad, _stride, _sum_col_data);
            ptr_1=_sum_col_data;
        } else {
            ptr_1=_sum_buffer;
        }


        int _a=n/4;
        int _b=n%4;

        for(int i=0;i<k;i++) {
            ptr_mlp=_mlp_buffer;

            for(int inc=0;inc<_a;inc++) {
                *ptr_mlp+=*ptr_1;
                *(ptr_mlp+1)+=*(ptr_1+1);
                *(ptr_mlp+2)+=*(ptr_1+2);
                *(ptr_mlp+3)+=*(ptr_1+3);

                ptr_1+=4;
                ptr_mlp+=4;
            }

            for(int inc=0;inc<_b;inc++) {
                *(ptr_mlp++)+=*(ptr_1++);
            }

        }

        ptr_mlp=_mlp_buffer;

        for(int inc=0;inc<_a;inc++) {
            *(ptr_mlp)=*(ptr_mlp)*t;
            *(ptr_mlp+1)=*(ptr_mlp+1)*t;
            *(ptr_mlp+2)=*(ptr_mlp+2)*t;
            *(ptr_mlp+3)=*(ptr_mlp+3)*t;

            ptr_1+=4;
            ptr_mlp+=4;
        }

        for(int inc=0;inc<_b;inc++) {
            *(ptr_mlp++)=*(ptr_mlp++)*t;
        }


    }

    unsigned char mpH[300] __attribute__ ((aligned (8)));

    void ConvolutionLayer::gemm_3X3_bit(int col_height,int col_width,int col_channel,int output_num,float *data_col,float *weight_data,float *output_data) {
        unsigned char *d=(unsigned char *)data_col;
        unsigned char *w=(unsigned char *)weight_data;
        float *o=output_data;

        int data_size=col_width*col_height;
        int output_data_size=_output[0]->count(2);
        int _n=data_size/4;
        int _m=data_size%4;

        int _k=col_channel/4;
        int _r=col_channel%4;

        register unsigned char tmp_kernel_1;
        register unsigned char tmp_kernel_2;
        register unsigned char tmp_kernel_3;
        register unsigned char tmp_kernel_4;


        register unsigned char tmp_data_1;
        register unsigned char tmp_data_2;
        register unsigned char tmp_data_3;
        register unsigned char tmp_data_4;


        for(int kernel_num=0;kernel_num<output_num;kernel_num++) {

            d=(unsigned char *)data_col;

            for(int inc=0;inc<_k;inc++) {

                o=output_data+kernel_num*output_data_size;

                tmp_kernel_1=*w;
                tmp_kernel_2=*(w+1);
                tmp_kernel_3=*(w+2);
                tmp_kernel_4=*(w+3);


                for(int step=0;step<_n;step++) {

                    tmp_data_1=*d;
                    tmp_data_1=*(d+1);
                    tmp_data_2=*(d+2);
                    tmp_data_3=*(d+3);

                    *(o)+=mpH[tmp_kernel_1&tmp_data_1]+mpH[tmp_kernel_2&tmp_data_1]+mpH[tmp_kernel_3&tmp_data_1]+mpH[tmp_kernel_4&tmp_data_1];
                    *(o+1)+=mpH[tmp_kernel_1&tmp_data_2]+mpH[tmp_kernel_2&tmp_data_2]+mpH[tmp_kernel_3&tmp_data_2]+mpH[tmp_kernel_4&tmp_data_2];
                    *(o+2)+=mpH[tmp_kernel_1&tmp_data_3]+mpH[tmp_kernel_2&tmp_data_3]+mpH[tmp_kernel_3&tmp_data_3]+mpH[tmp_kernel_4&tmp_data_3];
                    *(o+3)+=mpH[tmp_kernel_1&tmp_data_4]+mpH[tmp_kernel_2&tmp_data_4]+mpH[tmp_kernel_3&tmp_data_4]+mpH[tmp_kernel_4&tmp_data_4];

                    o+=4;
                    d+=4;
                }

                for(int step=0;step<_m;step++) {
                    tmp_data_1=*d;
                    *(o++)=mpH[tmp_kernel_1&tmp_data_1]+mpH[tmp_kernel_2&tmp_data_1]+mpH[tmp_kernel_3&tmp_data_1]+mpH[tmp_kernel_4&tmp_data_1];
                    d++;
                }

                w+=4;
            }

            for(int inc=0;inc<_r;inc++) {

                o=output_data+kernel_num*output_data_size;

                tmp_kernel_1=*w;

                for(int step=0;step<_n;step++) {

                    *(o)+=mpH[tmp_kernel_1&(*d)];
                    *(o+1)+=mpH[tmp_kernel_1&(*(d+1))];
                    *(o+2)+=mpH[tmp_kernel_1&(*(d+2))];
                    *(o+3)+=mpH[tmp_kernel_1&(*(d+3))];

                    o+=4;
                    d+=4;
                }

                for(int step=0;step<_m;step++) {
                    *(o++)=mpH[tmp_kernel_1&(*d++)];
                }

                w++;
            }
        }

        int _u=output_data_size/4;
        int _v=output_data_size%4;

        register float *ptr_o=output_data;
        register float *ptr_m=_mlp_buffer;

        for(int inc=0;inc<_u;inc++) {

            (*ptr_o)*=(*ptr_m);
            *(ptr_o+1)*=*(ptr_m+1);
            *(ptr_o+2)*=*(ptr_m+2);
            *(ptr_o+3)*=*(ptr_m+3);

            ptr_o+=4;
            ptr_m+=4;
        }

        for(int inc=0;inc<_v;inc++) {
            (*ptr_o++)*=(*ptr_m++);
        }
    }



    void ConvolutionLayer::gemm_1X1_() {

        int input_channels = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);

        int output_channels=_output[0]->dimension(1);
        int output_height = _output[0]->dimension(2);
        int output_width = _output[0]->dimension(3);


        float *input_data=_input[0]->get_data();
        float *d=input_data;
        float *w=_weight[0]->get_data();
        float *output_data=_output[0]->get_data();
        float *o=output_data;


        int data_size=input_height*input_width;
        int output_data_size=output_height*output_width;
        int _n=data_size/4;
        int _m=data_size%4;

        register float tmp_kernel_1;
        register float tmp_kernel_2;
        register float tmp_kernel_3;
        register float tmp_kernel_4;

        register float tmp_data_1;
        register float tmp_data_2;
        register float tmp_data_3;
        register float tmp_data_4;

        int _k=input_channels/4;
        int _r=input_channels%4;

        for(int kernel_num=0;kernel_num<output_channels;kernel_num++) {

            d=input_data;

            for(int inc=0;inc<_k;inc++) {

                o=output_data+kernel_num*output_data_size;

                tmp_kernel_1=*w;
                tmp_kernel_2=*(w+1);
                tmp_kernel_3=*(w+2);
                tmp_kernel_4=*(w+3);

                for(int step=0;step<_n;step++) {

                    tmp_data_1=*d;
                    tmp_data_2=*(d+1);
                    tmp_data_3=*(d+2);
                    tmp_data_4=*(d+3);

                    *(o)+=tmp_kernel_1*tmp_data_1+tmp_kernel_2*tmp_data_1+tmp_kernel_3*tmp_data_1+tmp_kernel_4*tmp_data_1;
                    *(o+1)+=tmp_kernel_1*tmp_data_2+tmp_kernel_2*tmp_data_2+tmp_kernel_3*tmp_data_2+tmp_kernel_4*tmp_data_2;
                    *(o+2)+=tmp_kernel_1*tmp_data_3+tmp_kernel_2*tmp_data_3+tmp_kernel_3*tmp_data_3+tmp_kernel_4*tmp_data_3;
                    *(o+3)+=tmp_kernel_1*tmp_data_4+tmp_kernel_2*tmp_data_4+tmp_kernel_3*tmp_data_4+tmp_kernel_4*tmp_data_4;

                    o+=4;
                    d+=4;
                }

                for(int step=0;step<_m;step++) {
                    tmp_data_1=*d;
                    *(o++)=tmp_kernel_1*tmp_data_1+tmp_kernel_2*tmp_data_1+tmp_kernel_3*tmp_data_1+tmp_kernel_4*tmp_data_1;
                    d++;
                }

                w+=4;
            }


            for(int inc=0;inc<_r;inc++) {

                o=output_data+kernel_num*output_data_size;

                tmp_kernel_1=*w;

                for(int step=0;step<_n;step++) {

                    *(o)+=tmp_kernel_1*(*d);
                    *(o+1)+=tmp_kernel_1*(*(d+1));
                    *(o+2)+=tmp_kernel_1*(*(d+2));
                    *(o+3)+=tmp_kernel_1*(*(d+3));

                    o+=4;
                    d+=4;
                }

                for(int step=0;step<_m;step++) {
                    *(o++)=tmp_kernel_1*(*d++);
                }

                w++;
            }
        }

    }



    void ConvolutionLayer::forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num) {

        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);

        //try im2col
        float *col_data = input_data;
        if (_need_im2col) {
            col_data = _col_buffer->get_data();
            im2col(input_data, input_channel, input_height, input_width, _kernel_size, _pad, _stride, col_data);
        }



            // Compute_Sum(input_data,input_height,input_width,input_channel);
            // Compute_Mlp(input_height,input_width);

        int m = _output[0]->dimension(1);  // output channel
        int n = _output[0]->count(2);  // output  width * height
        int k = _weight[0]->count(1);   // input_channels * kernel_h * kernel_w
        int weight_offset_ = m * k / _group; //get the kernel weight value
        int col_offset_ = k * n;        //get the "image" data in the col buffer
        int output_offset_ = m * n / _group; //get output data in the output buffer
        int pid = 1;
        for (int i = 0; i < _group; ++i) {

    /* sgemm("output channel",
            "output  width * height",
            "input_channels * kernel_h * kernel_w",
            "conv kernel data", //output_channel * (kernel_h*kernel_w*input_channels)
            "input data",       // (input_channel*kernel_h*kernel_w) * (output_width*output_height)
            "output data")  //output_channel * (output_h*output_w)
            */


                // dgemm_nn(m/_group,n,k/64,(unsigned int*)(weight_data + weight_offset_ * i),
                //             k/64,1,(unsigned int*)(col_data + col_offset_ * i),n,1,
                //                 output_data + output_offset_ * i,n,1);

            //    Gemmer::gemmers[pid - 1]->sgemm(m / _group,n,k,weight_data + weight_offset_ * i,
            //                                         col_data + col_offset_ * i, output_data + output_offset_ * i);

            //    int col_width = n;
            //    int col_channel=input_channel;
            //    int col_height = 1;

            //    gemm_3X3_bit(col_height,col_width,col_channel,
            //             m,col_data + col_offset_ * i,weight_data + weight_offset_ * i,output_data + output_offset_ * i);

            //    gemm_1X1_();


            if(_kernel_size==1) {

            } else if(_kernel_size==4) {
                
            }

        }

    }

#endif






    void ConvolutionLayer::forward_bias(float *output_data, float *bias_data, int thread_num) {
        int m = _output_num;
        int n = _output[0]->count(2);
        int k = 1;

        int pid = 1;

        Gemmer::gemmers[pid - 1]->sgemm(m, n, k, bias_data, _bias_buffer->get_data(), output_data, 1.0, 1.0);
    }

};
