
#include "layer/scale_layer.h"

namespace mdl {
    ScaleLayer::ScaleLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::SCALE;
        auto &param = config["param"];
        _bias_term = param["bias_term"].int_value();
        if (_bias_term > 0) {
            _bias_layer = new BiasLayer(config);

        }
        _scale = _weight[0];

        _outer_dim = _input[0]->count(0, 1);
        _scale_dim = _scale->count();
        _inner_dim = _input[0]->count(1 + _scale->get_dimensions().size());

    }

    ScaleLayer::~ScaleLayer() {
        if (_bias_layer != nullptr) {
            delete _bias_layer;
            _bias_layer = nullptr;
        }
    }

    void ScaleLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        for (int n = 0; n < _outer_dim; ++n) {
            for (int d = 0; d < _scale_dim; ++d) {
                float factor = _scale->get_data()[d];
                Math::v_scale(_inner_dim, factor, input_data, output_data);
                input_data += _inner_dim;
                output_data += _inner_dim;

            }

        }

        if (_bias_layer) {
            _bias_layer->forward();

        }
    }
}
