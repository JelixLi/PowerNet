
#ifndef Power_POOLING_LAYER_H
#define Power_POOLING_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class PoolingLayer: public Layer {
        public:
            PoolingLayer(const Json &config);
            void forward(int thread_num);
            void forward_max();
            void forward_ave();
        private:
            int _kernel_size;
            int _pad;
            int _stride;
            string _type;
            bool _global_pooling;
    };
};

#endif
