
#include "layer/convolution_layer.h"
#include "math/gemm.h"
#include <fstream>
#include <thread>

#ifdef OPTIMIZE

namespace mdl {
    ConvolutionLayer::ConvolutionLayer(const Json &config) : Layer(config), _col_buffer(nullptr),
                                                             _bias_buffer(nullptr) {

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

        _bias_buffer = new Matrix();
        _bias_buffer->resize({_output[0]->count(2)});
        _bias_buffer->reallocate();
        std::fill(_bias_buffer->get_data(), _bias_buffer->get_data() + _bias_buffer->count(), 1.0);
    }

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

    }

    typedef struct FLOAT_SINGLE
    {
        unsigned int nFraction3 : 8;
        unsigned int nFraction2 : 8;
        unsigned int nFraction1 : 7;
    
        unsigned int nExponent :  8;
    
        unsigned int nSign     :  1;
    
    } FP_SINGLE;
    

    void ConvolutionLayer::chan_gemm(float *input_data,bool *sign_data,int locx,int locy,int kernel_h,int kernel_w,float *weight_data,float &sum)
    {
        int input_width = _input[0]->dimension(3);
        int input_height = _input[0]->dimension(2);

        for(int i=0;i<kernel_h;i++)
            for(int j=0;j<kernel_w;j++)
            {   
                int x=locx+i;
                int y=locy+j;
                
                if(x>=0&&y>=0&&x<input_height&&y<input_width)
                {
                    int input_offset=x*input_width+y;
    
                    int weight_offset=i*kernel_w+j;
                    
                    // float t_1=input_data[input_offset];
                    // float w=weight_data[weight_offset];

                    // FP_SINGLE *fp=(FP_SINGLE*)&t_1;
                    // unsigned char old=fp->nExponent;
                    // fp->nExponent+=w;
                    // unsigned char s_new=fp->nExponent;

                    // if(w<0&&s_new>old)
                    //     fp->nExponent=1;
                    // else if(w>0&&s_new<old)
                    //     fp->nExponent=254;

                    // if(sign_data[weight_offset])
                    // {
                    //     sum+=t_1;
                    // }
                    // else
                    // {
                    //     sum-=t_1;
                    //     t_1=t_1*-1;
                    // }

                    sum+=input_data[input_offset]*weight_data[weight_offset];
                }

            }

    }

    void ConvolutionLayer::Power_Gemm(float *input_data,float *weight_data,float *output_data)
    {

        int input_channel = _input[0]->dimension(1)/_group;
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);

        int output_channel = _output[0]->dimension(1)/_group;
        int output_height = _output[0]->dimension(2);
        int output_width = _output[0]->dimension(3);

        int kernel_h = _weight[0]->dimension(2);
        int kernel_w = _weight[0]->dimension(3);

        bool *sign_data = get_sign()[0]->get_sign_data();

        int image_size = input_height*input_width;
        int kernel_size = kernel_w*kernel_h;

        int output_offset=0;
        int padding = _pad;
        for(int i=0;i<output_channel;i++)
        {
            for(int locx=padding*-1;locx<=input_height+padding-kernel_h;locx+=_stride)
                for(int locy=padding*-1;locy<=input_width+padding-kernel_w;locy+=_stride)
                {
                    float *p=input_data;
                    float *w=weight_data;
                    bool *s=sign_data;
                    float sum=0;
                    for(int cnt=0;cnt<input_channel;s+=kernel_size,p+=image_size,w+=kernel_size,cnt++)
                    {
                        chan_gemm(p,s,locx,locy,kernel_w,kernel_h,w,sum);
                    }
                    
                    output_data[output_offset++]=sum;
                }

                weight_data+=kernel_size*input_channel;
                sign_data+=kernel_size*input_channel;

        }

    }

    void ConvolutionLayer::forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num) {

        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);

        int m = _output[0]->dimension(1);  // output channel
        int n = _output[0]->count(2);  // output  width * height
        int k = _weight[0]->count(1);   // input_channels * kernel_h * kernel_w
        int weight_offset_ = m * k / _group; //get the kernel weight value
        int output_offset_ = m * n / _group; //get output data in the output buffer
        int input_offset = input_height*input_width*input_channel/_group;


        for(int i=0;i<_group;i++)
        {
            Power_Gemm(input_data+i*input_offset,weight_data+i*weight_offset_,output_data+i*output_offset_);
        }

    }


    void ConvolutionLayer::forward_bias(float *output_data, float *bias_data, int thread_num) {
        int m = _output_num;
        int n = _output[0]->count(2);
        int k = 1;

        int pid = 1;

        Gemmer::gemmers[pid - 1]->sgemm(m, n, k, bias_data, _bias_buffer->get_data(), output_data, 1.0, 1.0);
    }

};



#else


namespace mdl {
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
    }

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

    void ConvolutionLayer::forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num) {
        int input_channel = _input[0]->dimension(1);
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);
        // try im2col
        float *col_data = input_data;
        if (_need_im2col) {
            col_data = _col_buffer->get_data();
            im2col(input_data, input_channel, input_height, input_width, _kernel_size, _pad, _stride, col_data);
        }
        int m = _output[0]->dimension(1);  // output channel
        int n = _output[0]->count(2);  // output  width * height
        int k = _weight[0]->count(1);   // input_channels * kernel_h * kernel_w
        int weight_offset_ = m * k / _group; //get the kernel weight value
        int col_offset_ = k * n;        //get the "image" data in the col buffer
        int output_offset_ = m * n / _group; //get output data in the output buffer

#ifdef MULTI_THREAD
        if (thread_num > 1) {
            std::thread *ths = new std::thread[thread_num];
            int m1 = m / thread_num;
            int m2 = m % thread_num;
            for (int i = 0; i < thread_num; i++) {
                int row_count = m1;
                if (i == thread_num - 1) {
                    row_count = m1 + m2;
                }
                ths[i] = std::thread(run1, i, row_count, n, k, weight_data + i * m1 * k, col_data,
                                     output_data + i * m1 * n);
            }
            for (int j = 0; j < thread_num; ++j) {
                ths[j].join();

            }
            delete []ths;
            return;
        }

        int pid = this->pid();
        if (pid < 1 || pid > Gemmer::gemmers.size()) {
            pid = 1;
        }
#else
        int pid = 1;
#endif

        for (int i = 0; i < _group; ++i) {

    /* sgemm("output channel",
            "output  width * height",
            "input_channels * kernel_h * kernel_w", 
            "conv kernel data", //output_channel * (kernel_h*kernel_w*input_channels)
            "input data",       // (input_channel*kernel_h*kernel_w) * (output_width*output_height)
            "output data")  //output_channel * (output_h*output_w)
            */
                                         
            Gemmer::gemmers[pid - 1]->sgemm(m / _group,n,k,weight_data + weight_offset_ * i,
                                            col_data + col_offset_ * i, output_data + output_offset_ * i);
                                     
        }

    }


    void ConvolutionLayer::forward_bias(float *output_data, float *bias_data, int thread_num) {
        int m = _output_num;
        int n = _output[0]->count(2);
        int k = 1;

#ifdef MULTI_THREAD
        if (thread_num > 1) {
            std::thread *ths = new std::thread[thread_num];
            int m1 = m / thread_num;
            int m2 = m % thread_num;
            for (int i = 0; i < thread_num; i++) {
                int row_count = m1;
                if (i == thread_num - 1) {
                    row_count = m1 + m2;
                }
                ths[i] = std::thread(run2, i, row_count, n, k, bias_data + i * m1 * k, _bias_buffer->get_data(),
                                     output_data + i * m1 * n, 1.0, 1.0);
            }
            for (int j = 0; j < thread_num; ++j) {
                ths[j].join();

            }
            delete []ths;
            return;
        }

        int pid = this->pid();
        if (pid < 1 || pid > Gemmer::gemmers.size()) {
            pid = 1;
        }
#else
        int pid = 1;
#endif

        Gemmer::gemmers[pid - 1]->sgemm(m, n, k, bias_data, _bias_buffer->get_data(), output_data, 1.0, 1.0);
    }

};

#endif