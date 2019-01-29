
#include "base/matrix.h"

namespace Power {
    Matrix::Matrix(): 
        _data(nullptr), _data_size(0), _dimensions_size(0), _is_external_memory(false),sign_data(nullptr) {
    }

    Matrix::Matrix(const Json &config):
        _data(nullptr), _data_size(0), _dimensions_size(0), _is_external_memory(false),sign_data(nullptr) {
        vector<int> dimensions;

        for (int index = 0; index < config.array_items().size(); index++) {
            dimensions.push_back(config[index].int_value());
        }

        resize(dimensions);
    }
    
    Matrix::~Matrix() {
        clear_data();
    }


    void Matrix::resize(const vector<int> &dimensions) {
        _dimensions = dimensions;
        _dimensions_size = dimensions.size();
        _data_size = count(0);
    }

    void Matrix::reallocate(float value) {
        clear_data();
        _is_external_memory = false;
        _data = new float[_data_size];
        std::fill(_data, _data + _data_size, value);
    }

    void Matrix::clear_data() {
        if (_data != nullptr) {
            if (!_is_external_memory) {
                delete[] _data;
            }
            _data = nullptr;
        }
        if (sign_data != nullptr) {
            if (!_is_external_memory) {
                delete[] sign_data;
            }
            sign_data = nullptr;
        }
    }
};
