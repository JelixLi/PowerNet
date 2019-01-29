
#ifndef Power_SPLIT_LAYER_H
#define Power_SPLIT_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class SplitLayer: public Layer {
        public:
            SplitLayer(const Json &config);
            void forward(int thread_num);
    };
};

#endif
