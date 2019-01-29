

#include "layer/sigmoid_layer.h"

namespace Power {
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
            for (int i=0; i<size; i++)
            {
                output_data[i] = 1.f / (1.f + exp(-input_data[i]));
            }
            input_data += _input[0]->offset({0, 1});
            output_data += _output[0]->offset({0, 1});
        }


    }
}
