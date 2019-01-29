
#ifndef Power_CONCAT_LAYER_H
#define Power_CONCAT_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class ConcatLayer: public Layer {
        public:
            ConcatLayer(const Json &config);
            void forward(int thread_num);
    };
};

#endif
