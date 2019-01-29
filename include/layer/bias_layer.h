
#ifndef Power_BIAS_LAYER_H
#define Power_BIAS_LAYER_H

#include "base/layer.h"

namespace Power {
    class BiasLayer:public Layer {
    public:
        BiasLayer(const Json &jsonValue);
        ~BiasLayer();
        void forward();
    private:
        Matrix *_bias_multiplier;
        Matrix *_bias;
        int _outer_dim, _bias_dim, _inner_dim, _dim;
    };
}

#endif //Power_BIAS_LAYER_H
