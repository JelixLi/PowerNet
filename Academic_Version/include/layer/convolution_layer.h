
#ifndef MDL_CONVOLUTION_LAYER_H
#define MDL_CONVOLUTION_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace mdl {
    class ConvolutionLayer: public Layer {
        public:
            ConvolutionLayer(const Json &config);
            ~ConvolutionLayer();
            void forward(int thread_num);

            int get_kernel_size()
            {
                return _kernel_size;
            }

#ifdef XNOR_MODE

            void Compute_Mlp();

            void Compute_Sum(float *input_data,int input_height,int input_width,int input_channel);

            void Compute_Mlp(int input_height,int input_width);

            void gemm_3X3_bit(int col_height,int col_width,int col_channel,
                    int output_num,float *data_col,float *weight_data,float *output_data);

            void gemm_1X1_();

#endif

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

#ifdef XNOR_MODE
            //xnor mode
            float *_sum_buffer;
            float *_mlp_buffer;
            float *_mlp_kernel_buffer;
            float *_sum_col_data;
#endif 

#ifdef GPU_MODE
            //gpu mode
            float *_gpu_input_data_buffer;
            float *_gpu_output_data_buffer;
            float *_gpu_kernel_buffer;
#endif
    };
};

#endif


