
#ifndef Power_NET_H
#define Power_NET_H

#include "base/layer.h"
#include "layer/convolution_layer.h"

namespace Power {

    class Net {
    public:

        Net(const Json &config);

        ~Net();

        vector<float> predict(float *image);

        vector<float> forward_from_to(float *image, int start, int end, bool sampling = false);

        void set_thread_num(int thread_num) {
            _thread_num = thread_num;
        }


#ifdef NEED_DUMP

        void dump(string filename);

#endif



    private:
        string _name;

        int _thread_num;

        vector<Layer *> _layers;

#ifdef NEED_DUMP
        void dump_with_quantification(string filename);

        void dump_without_quantification(string filename);
#endif



    };
};

#endif
