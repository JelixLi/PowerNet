
#include "layer/fc_layer.h"

#include "math/gemm.h"

namespace Power {
    FCLayer::FCLayer(const Json &config): Layer(config) {
        assure_memory();
        auto &param = config["param"];
        _layer_type = LayerType::FULLCONNECT;
        _output_num = param["output_num"].int_value();
        _bias_buffer = new Matrix();
        _bias_buffer->resize({1, _input[0]->count(0, 1)});
        _bias_buffer->reallocate(1.0);
    }

    FCLayer::~FCLayer() {
        if (_bias_buffer != nullptr) {
            delete _bias_buffer;
            _bias_buffer = nullptr;
        }
    }

    void FCLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *weight_data = _weight[0]->get_data();
        float *bias_data = _weight[1]->get_data();
        int m = _input[0]->count(0, 1);
        int n = _output_num;
        int k = _input[0]->count(1);

        Gemmer::gemmers[0]->sgemm(m, n, k, input_data, weight_data, output_data);
        Gemmer::gemmers[0]->sgemm(m, n, 1, _bias_buffer->get_data(), bias_data, output_data, 1.0, 1.0);

        descript();
    }
};
