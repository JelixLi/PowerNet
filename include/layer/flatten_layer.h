

#ifndef Power_FLATTEN_LAYER_H
#define Power_FLATTEN_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class FlattenLayer: Layer {
    public:
        int flatten_start = 1;
        int flatten_end = -1;
        FlattenLayer(const Json &config);
        ~FlattenLayer();

        void forward(int thread_num);

    };
}
#endif //Power_FLATTEN_LAYER_H
