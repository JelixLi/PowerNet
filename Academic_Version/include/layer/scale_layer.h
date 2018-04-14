
#ifndef MOBILE_DEEP_LEARNING_SCALE_LAYER_H
#define MOBILE_DEEP_LEARNING_SCALE_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"
#include "layer/bias_layer.h"

namespace mdl {
    class ScaleLayer : public Layer {
    public:
        ScaleLayer(const Json &config);

        ~ScaleLayer();

        void forward(int thread_num);

        int _bias_term;
        Matrix * _scale;
        BiasLayer *_bias_layer;
        int _outer_dim, _scale_dim, _inner_dim;
    };
}
#endif //MOBILE_DEEP_LEARNING_SCALE_LAYER_H
