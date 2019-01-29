
#include "layer/concat_layer.h"

namespace Power {
    ConcatLayer::ConcatLayer(const Json &config): Layer(config) {
        assure_memory();
        _layer_type = LayerType::CONCAT;
    }

    void ConcatLayer::forward(int thread_num) {
        if (_input.size() == 1) {
            return ;
        }
        int offset = 0;
        int output_offset = _output[0]->dimension(1);
        float *output_data = _output[0]->get_data();
        int count = _input[0]->count(0, 1);
        int input_size = _input[0]->count(2);
        for (auto &src: _input) {
            int input_offset = src->dimension(1);
            float *input_data = src->get_data();
            for (int i = 0; i < count; i++) {
                float *input_data_start = input_data + i * input_offset * input_size;
                std::copy(input_data_start, input_data_start + input_offset * input_size, output_data + (i * output_offset + offset) * input_size);
            }
            offset += input_offset;
        }
        descript();
    }



};
