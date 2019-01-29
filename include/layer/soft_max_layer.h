

#ifndef Power_SOFT_MAX_LAYER_H
#define Power_SOFT_MAX_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class SoftmaxLayer : public Layer {
    public:
        SoftmaxLayer(const Json &config);

        ~SoftmaxLayer();

        void forward(int thread_num);

        int _outer_count;
        int _inner_count;
        int _softmax_dim;
        Matrix*_sum_matrix;
        Matrix* _scale_matrix;

    };

}

#endif //Power_SOFT_MAX_LAYER_H
