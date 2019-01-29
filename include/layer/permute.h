

#ifndef Power_PERMUTE_H
#define Power_PERMUTE_H

#include "base/layer.h"
#include "commons/commons.h"
namespace Power {
    class PermuteLayer : Layer {
    public:
        PermuteLayer(const Json &config);
        ~PermuteLayer();
        void forward(int thread_num);

    };
}



#endif //Power_PERMUTE_H
