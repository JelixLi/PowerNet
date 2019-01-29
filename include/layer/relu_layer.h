
#ifndef Power_RELU_LAYER_H
#define Power_RELU_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class ReluLayer: public Layer {
        public:
            ReluLayer(const Json &config);
    		void forward(int thread_num);
    };
};

#endif
