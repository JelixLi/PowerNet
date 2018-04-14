

#ifndef MOBILE_DEEP_LEARNING_SOFT_MAX_LAYER_H
#define MOBILE_DEEP_LEARNING_SOFT_MAX_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace mdl {
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

#endif //MOBILE_DEEP_LEARNING_SOFT_MAX_LAYER_H
