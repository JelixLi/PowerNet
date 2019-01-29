
#include "layer/convolution_layer.h"
#include "math/gemm.h"
#include <fstream>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <string.h>

#ifdef GPU
#include "math/gpu_gemm.h" 
#endif 

namespace Power {

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

// #ifdef GPU 
//         float *bias_data = _weight[1]->get_data();
//         global_gpu_manager.TransBiasToGpuFormat(_bias_buffer,bias_data,_output_num,_output[0]->count(2));
// #endif 

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
            forward_gemm(input_data + i * input_offset, weight_data, output_data + i * output_offset,thread_num);
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

        //try im2col
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
        int pid = 1;



#ifndef GPU
        for (int i = 0; i < _group; ++i) {
           Gemmer::gemmers[pid - 1]->sgemm(m / _group,n,k,weight_data + weight_offset_ * i,
                                                col_data + col_offset_ * i, output_data + output_offset_ * i);
        }
#else 
        Power::global_gpu_manager.gpu_conv(
                          weight_data,
                          col_data,
                          output_data,
                          input_height,
                          input_width,
                          input_channel,
                          _kernel_size,
                          _output_num,
                          _pad,
                          _stride);
#endif



    }


    void ConvolutionLayer::forward_bias(float *output_data, float *bias_data, int thread_num) {
        int m = _output_num;
        int n = _output[0]->count(2);
        int k = 1;

        int pid = 1;

        Gemmer::gemmers[pid - 1]->sgemm(m, n, k, bias_data, _bias_buffer->get_data(), output_data, 1.0, 1.0);
    }

};
