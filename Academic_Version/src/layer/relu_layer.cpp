
#include "layer/relu_layer.h"

namespace mdl {
    ReluLayer::ReluLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::RELU;
        _pid = config["pid"].int_value();
    }

    void ReluLayer::forward(int thread_num) {
        float *src = _input[0]->get_data();
        float *dest = _output[0]->get_data();
        int count = _input[0]->count();
        for (int i = 0; i < count; i++) {
            dest[i] = std::max(src[i], 0.0f);
        }
        descript();
    }
};
