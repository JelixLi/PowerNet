
#ifndef Power_ELT_WISE_LAYER_H
#define Power_ELT_WISE_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"

namespace Power {
    class EltWiseLayer : public Layer {
    public:
        EltWiseLayer(const Json &config);
        ~EltWiseLayer();
        void forward(int thread_num);
        void forward_sum();
        void forward_max();
        void forward_product();

        string _type;
        vector<float>_coeffs;

    };
}




#endif //Power_ELT_WISE_LAYER_H
