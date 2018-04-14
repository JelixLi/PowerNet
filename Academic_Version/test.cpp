
#include "layer/convolution_layer.h"
#include "math/gemm.h"
#include <fstream>
#include <thread>

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
    

    void chan_gemm(float *input_data,bool *sign_data,int locx,int locy,float *weight_data,int &sum)
    {
        for(int i=0;i<kernel_h;i++)
            for(int j=0;j<kernel_w;j++)
            {
                int x=locx+i;
                int y=locy+j;
                int input_offset=x*input_height+y;
                int weight_offset=i*kernel_h+j;

                FP_SINGLE *fp=&input_data[input_offset];
                fp->nExponent+=weight_data[weight_offset];
                if(sign_data)
                    sum+=input_data[input_offset];
                else
                    sum-=input_data[input_offset];

            }

    }

    void Power_Gemm(float *input_data,float *weight_data,float *output_data)
    {

        int input_channel = _input[0]->dimension(1)/_group;
        int input_height = _input[0]->dimension(2);
        int input_width = _input[0]->dimension(3);

        int image_size = input_height*input_width;

        int output_channel = _output[0]->dimension(1)/_group;
        int output_height = _output[0]->dimension(2);
        int output_width = _output[0]->dimension(3);

        int kernel_h = _weight[0]->dimension(2);
        int kernel_w = _weight[0]->dimension(3);

        int output_offset=0;
        for(int i=0;i<output_channel;i++)
        {
            weight_data += kernel_h*kernel_w;
            for(int locx=0;locx<input_height;locx+=_stride)
                for(int locy=0;locy<input_width;locy+=_stride)
                {
                    float *p=input_data;
                    bool *s=sign_data;
                    int sum=0;
                    for(int cnt=0;cnt<input_channel;s+=image_size,p+=image_size,cnt++)
                    {
                        chan_gemm(p,s,locx,locy,kernel_w,kernel_h,weight_data,sum);
                    }
                    
                    output_data[output_offset++]=sum;
                }


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

