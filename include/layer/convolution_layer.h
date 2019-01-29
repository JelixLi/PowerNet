
#ifndef Power_CONVOLUTION_LAYER_H
#define Power_CONVOLUTION_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class ConvolutionLayer: public Layer {
        public:
            ConvolutionLayer(const Json &config);
            ~ConvolutionLayer();
            void forward(int thread_num);

            int get_kernel_size()
            {
                return _kernel_size;
            }


        private:
            void forward_gemm(float *input_data, float *weight_data, float *output_data, int thread_num);
            void forward_bias(float *output_data, float *bias_data, int thread_num);
            int _output_num;
            int _kernel_size;
            int _pad;
            int _stride;
            int _bias_term;
            int _group;
            bool _need_im2col;
            Matrix *_col_buffer;
            Matrix *_bias_buffer;
            float *_sum_buffer;
            float *_mlp_buffer;
            float *_mlp_kernel_buffer;
            float *_sum_col_data;
    };
};

#endif
