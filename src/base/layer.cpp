
#include "base/layer.h"
#include "loader/loader.h"

namespace Power {

    Layer::Layer(const Json &config) {
        _name = config["name"].string_value();

        string matrix_name = "";
        for (int i = 0; i < config["input"].array_items().size(); ++i) {
            matrix_name = config["input"][i].string_value();
            auto matrix = Loader::shared_instance()->_matrices[matrix_name];
            if (matrix) {
                matrix->set_name(matrix_name);
                _input.push_back(matrix);
            }
        }
        for (int i = 0; i < config["output"].array_items().size(); ++i) {
            matrix_name = config["output"][i].string_value();
            auto matrix = Loader::shared_instance()->_matrices[matrix_name];
            if (matrix) {
                matrix->set_name(matrix_name);
                _output.push_back(matrix);
            }
        }
        for (int i = 0; i < config["weight"].array_items().size(); ++i) {
            matrix_name = config["weight"][i].string_value();

            auto matrix = Loader::shared_instance()->_matrices[matrix_name];

            if (matrix) {
                matrix->set_name(matrix_name);
                _weight.push_back(matrix);
            }
        }
    }
};
