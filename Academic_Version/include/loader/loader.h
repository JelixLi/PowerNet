
#ifndef MDL_LOADER_H
#define MDL_LOADER_H

#include "commons/commons.h"

namespace mdl {
    class Matrix;

    /**
     * use loader object to load net description file & net params
     */
    class Loader {
    public:
        /**
         * get single instance of loader
         * @return Loader object
         */
        static Loader *shared_instance();

        /**
         * json object for the json file of net description
         */
        Json _model;

        map<string, Matrix *> _matrices;

        /**
         * load net description and params
         * clear should be called before reload, or an exception will be thrown
         * @param model_path
         * @param weights_path
         * @return true if load success
         */
        bool load(string model_path, string weights_path);

        /**
         * clear all data, ensure that it be called before load again
         */
        void clear();

        /**
         * check wheather the model and params are loaded
         * @return true or false
         */
        bool get_loaded() {
            return _loaded;
        }


    private:
        bool _loaded;

        bool _cleared;

        static Loader *_instance;

        Loader() : _loaded(false), _cleared(true) {}

        /**
         * load the json file of net description and params
         * note that the params are quantified in order to compress the size of params
         * quanification converts the type of params from float to int, which decreases the size to one quarter of
         * the original
         * @param model_path
         * @param weights_path
         * @return true or false
         */
        bool load_with_quantification(string model_path, string weights_path);


#ifdef NEED_DUMP
        /**
         * load without quantification
         * @param model_path
         * @param weights_path
         * @return
         */
        bool load_without_quantification(string model_path, string weights_path);
#endif
    };
};

#endif
