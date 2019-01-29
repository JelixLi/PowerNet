
#ifndef Power_LRN_LAYER_H
#define Power_LRN_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class LrnLayer: public Layer {
        public:
            LrnLayer(const Json &config);
            ~LrnLayer();
            void forward(int thread_num);
        private:
            float _alpha;
            float _beta;
            int _local_size;
            Matrix *_scale_buffer;
            Matrix *_sqr_buffer;
    };
};

#endif
