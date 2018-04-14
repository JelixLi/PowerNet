

#include "layer/flatten_layer.h"

namespace mdl {
    FlattenLayer::FlattenLayer(const Json &config) : Layer(config) {
        auto &param = config["param"];
        flatten_start = param["start"].int_value();
        int input_dimensions = _input[0]->get_dimensions().size();
        flatten_end = param["end"].int_value();
        if (flatten_end < 0) {
        flatten_end +=input_dimensions;
        }
        vector<int> output_shape;
        for (int i = 0; i < flatten_start; ++i) {
            output_shape.push_back(_input[0]->dimension(i));
        }
        int flatten_size = _input[0]->count(flatten_start, flatten_end);
        for (int j = flatten_end + 1; j < input_dimensions; ++j) {
            output_shape.push_back(_input[0]->dimension(j));

        }
        _output[0]->resize(output_shape);

    }

    FlattenLayer::~FlattenLayer() {}

    void FlattenLayer::forward(int thread_num) {
        _output[0]->set_data(_input[0]->get_data());

    }
}