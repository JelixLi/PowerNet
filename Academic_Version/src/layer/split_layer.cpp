
#include "layer/split_layer.h"

namespace mdl {
    SplitLayer::SplitLayer(const Json &config): Layer(config) {
        assure_memory();
        _layer_type = LayerType::SPLIT;
    }

    void SplitLayer::forward(int thread_num) {
        auto source = _input[0];
        for (auto matrix: _output) {
            matrix->set_data(source->get_data());
        }
        descript();
    }
};
