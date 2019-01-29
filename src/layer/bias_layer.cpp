
#include "layer/bias_layer.h"
#include "math/gemm.h"

namespace Power {
    BiasLayer::BiasLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::BIAS;
        int axis = 1;
        int num_axis = 1;
        auto begin = _input[0]->get_dimensions().begin();
        vector<int> bias_shape(begin + 1, begin + 2);
        _bias = _weight[1];
        _outer_dim = _input[0]->count(0, axis);
        _bias_dim = _bias->count();
        _inner_dim = _input[0]->count(axis + _bias->get_dimensions().size());
        _dim = _bias_dim * _inner_dim;
        _bias_multiplier = new Matrix();
        _bias_multiplier->resize(vector<int>(1, _inner_dim));
        _bias_multiplier->reallocate(1);

    }

    BiasLayer::~BiasLayer() {
        if (_bias_multiplier != nullptr) {
            delete _bias_multiplier;
            _bias_multiplier = nullptr;
        }

    }

    void BiasLayer::forward() {
        float *output_data = _output[0]->get_data();
        float *bias_data = _bias->get_data();
        for (int i = 0; i < _outer_dim; ++i) {
            Gemmer::gemmers[0]->sgemm(_bias_dim, _inner_dim, 1, bias_data, _bias_multiplier->get_data(),
                                      output_data, 1, 1);
            output_data += _dim;

        }

    }




}

