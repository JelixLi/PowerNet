
#ifndef MDL_RELU_LAYER_H
#define MDL_RELU_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace mdl {
    class ReluLayer: public Layer {
        public:
            ReluLayer(const Json &config);
            void forward(int thread_num);
    };
};

#endif
