
#ifndef Power_FC_LAYER_H
#define Power_FC_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class FCLayer: public Layer {
        public:
            FCLayer(const Json &config);
            ~FCLayer();
            void forward(int thread_num);
        private:
            int _output_num;
            Matrix *_bias_buffer;
    };
};

#endif
