
#include "layer/soft_max_layer.h"
#include "math/gemm.h"

namespace mdl {
    SoftmaxLayer::SoftmaxLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::SOFTMAX;
        _softmax_dim = 1;
        vector<int> dimensions(1, _input[0]->dimension(1));
        _sum_matrix = new Matrix();
        _sum_matrix->resize(dimensions);
        _sum_matrix->reallocate(1);
        _outer_count = _input[0]->count(0, _softmax_dim);
        _inner_count = _input[0]->count(_softmax_dim + 1);
        vector<int> scale_dims = _input[0]->get_dimensions();
        scale_dims[_softmax_dim] = 1;

        _scale_matrix = new Matrix();
        _scale_matrix->resize(scale_dims);
        _scale_matrix->reallocate();
    }

    SoftmaxLayer::~SoftmaxLayer() {
        if (_sum_matrix != nullptr) {
            delete _sum_matrix;
            _sum_matrix = nullptr;
        }
        if (_scale_matrix != nullptr) {
            delete _scale_matrix;
            _scale_matrix = nullptr;
        }

    }

    void SoftmaxLayer::forward(int thread_num) {
        float *input_data = _input[0]->get_data();
        float *output_data = _output[0]->get_data();
        float *scale_data = _scale_matrix->get_data();
        int channels = _input[0]->dimension(_softmax_dim);
        int dim = _input[0]->count() / _outer_count;
        copy(_input[0]->count(), input_data, output_data);
        for (int i = 0; i < _outer_count; ++i) {
            copy(_inner_count, input_data + i * dim, scale_data);
            for (int j = 0; j < channels; ++j) {
                for (int k = 0; k < _inner_count; ++k) {
                    scale_data[k] = std::max(scale_data[k],
                                             input_data[i * dim + j * _inner_count + k]);
                }

            }
            Gemmer::gemmers[0]->sgemm(channels, _inner_count, 1,
                                      _sum_matrix->get_data(), scale_data, output_data, -1., 1);

            Math::v_exp(dim, output_data, output_data);
            float sum = 0;
            for (int m = 0; m < channels; ++m) {
                sum += output_data[m] * _sum_matrix->get_data()[m];
            }
            scale_data[0] = sum;
            for (int l = 0; l < channels; ++l) {
                Math::v_div(_inner_count, output_data, scale_data, output_data);
                output_data += _inner_count;
            }
        }

    }
}
