
#ifndef MDL_SPLIT_LAYER_H
#define MDL_SPLIT_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace mdl {
    class SplitLayer: public Layer {
        public:
            SplitLayer(const Json &config);
            void forward(int thread_num);
    };
};

#endif
