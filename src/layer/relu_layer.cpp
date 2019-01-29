
#include "layer/relu_layer.h"

namespace Power {
    ReluLayer::ReluLayer(const Json &config) : Layer(config) {
        assure_memory();
        _layer_type = LayerType::RELU;
        _pid = config["pid"].int_value();
    }

    void ReluLayer::forward(int thread_num) {
        float *src = _input[0]->get_data();
        float *dest = _output[0]->get_data();
        int count = _input[0]->count();

        int _k=count/4;
        int _r=count%4;

        // for (int i = 0; i < count; i++) {
        //     dest[i] = std::max(src[i], 0.0f);
        // }

        register float *ptr_1=src;
        register float *ptr_2=dest;

        for (int i = 0; i < _k; i+=4) {
            *ptr_2=std::max(*ptr_1,0.0f);
            *(ptr_2+1)=std::max(*(ptr_1+1),0.0f);
            *(ptr_2+2)=std::max(*(ptr_1+2),0.0f);
            *(ptr_2+3)=std::max(*(ptr_1+3),0.0f);

            ptr_1+=4;
            ptr_2+=4;
        }

        for (int i = 0; i < _r; i++) {
            *ptr_2++=std::max(*ptr_1++,0.0f);
        }
        descript();
    }
};
