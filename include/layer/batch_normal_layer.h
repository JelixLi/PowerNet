
#ifndef Power_BATCH_NORMAL_LAYER_H
#define Power_BATCH_NORMAL_LAYER_H

#include "commons/commons.h"
#include "base/layer.h"
namespace Power {
    class BatchNormalLayer: public Layer {
    public:
        BatchNormalLayer(const Json &config);
        ~BatchNormalLayer();
        void forward(int thread_num);

        // means of each channels
        Matrix *_mean;

        // variance of each channels
        Matrix *_variance;

        // temp data matrix
        Matrix *_temp;

        int _channels;

        float _eps = 0.000010;

        // assist matrix for gemm
        Matrix *_batch_sum_multiplier;
        Matrix *_num_by_chans;
        Matrix *_spatial_sum_mutiplier;




    };
}



#endif //Power_BATCH_NORMAL_LAYER_H
