

#ifndef MOBILE_DEEP_LEARNING_SIGMOID_LAYER_H
#define MOBILE_DEEP_LEARNING_SIGMOID_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"
namespace mdl {
    class SigmoidLayer: public Layer {
    public:
        SigmoidLayer(const Json &config);
        ~SigmoidLayer();
        void forward(int thread_num);


    };
}

#endif //MOBILE_DEEP_LEARNING_SIGMOID_LAYER_H
