

#include "layer/sigmoid_layer.h"

namespace mdl {
    SigmoidLayer::SigmoidLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::SIGMOID;

    }

    SigmoidLayer::~SigmoidLayer() {

    }

    inline float sigmoid(float x) {
        return 1. / (1. + exp(-x));
    }


    void SigmoidLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        int channel = _input[0]->dimension(1);
        int size = _input[0]->count(2,3);
        for (int c = 0; c < channel; c++) {
#ifdef ANDROID
            int align = size / 4;
            int remain = size - (align * 4);

            float32x4_t _one = vdupq_n_f32(1.f);
            for (int i = 0; i < align; i++) {
                float32x4_t data = vld1q_f32(input_data);
                data = vnegq_f32(data);
                data = exp_ps(data);
                data = vaddq_f32(data, _one);
                float32x4_t out_data = vrecpeq_f32(data);
                out_data = vmulq_f32(vrecpsq_f32(data, out_data), out_data);
                vst1q_f32(output_data, out_data);

                input_data += 4;
                output_data += 4;
            }

            for (int i = 0; i < remain; i++) {
                *output_data = 1.f / (1.f + exp(-*input_data));
                output_data++;
                input_data++;
            }
            input_data += _input[0]->offset({0, 1});
            output_data += _output[0]->offset({0, 1});
#else
            for (int i=0; i<size; i++)
            {
                output_data[i] = 1.f / (1.f + exp(-input_data[i]));
            }
            input_data += _input[0]->offset({0, 1});
            output_data += _output[0]->offset({0, 1});
#endif
        }


    }
}
