
#ifndef MDL_NET_H
#define MDL_NET_H

#include "base/layer.h"

namespace mdl {
    /**
     * do object detection with Net object
     */
    class Net {
    public:
        /**
         * net init
         * @param config
         * @return
         */
        Net(const Json &config);

        ~Net();

        /**
         * object detection
         * @param image
         * @return vector of float
         */
        vector<float> predict(float *image);

        /**
         * forward
         * @param image
         * @param start
         * @param end
         * @param sampling
         * @return vector of float
         */
        vector<float> forward_from_to(float *image, int start, int end, bool sampling = false);

        /**
         * set the thread num
         * @param thread_num
         */
        void set_thread_num(int thread_num) {
            _thread_num = thread_num;
        }
        
        void Transform_Conv();

#ifdef NEED_DUMP

        void dump(string filename);

#endif
    private:
        string _name;

        int _thread_num;

        vector<Layer *> _layers;

        vector<Layer *> convptr;

#ifdef NEED_DUMP
        void dump_with_quantification(string filename);

        void dump_without_quantification(string filename);
#endif
    };
};

#endif
