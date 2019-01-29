

#ifndef Power_SIGMOID_LAYER_H
#define Power_SIGMOID_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"
namespace Power {
    class SigmoidLayer: public Layer {
    public:
        SigmoidLayer(const Json &config);
        ~SigmoidLayer();
        void forward(int thread_num);


    };
}

#endif //Power_SIGMOID_LAYER_H
