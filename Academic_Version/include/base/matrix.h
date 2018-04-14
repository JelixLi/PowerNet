
#ifndef MDL_MATRIX_H
#define MDL_MATRIX_H

#include <fstream>
#include "commons/commons.h"

namespace mdl {
    class Matrix {
    public:
        Matrix();

        /**
         * construct fun
         * @param config
         * @return
         */
        Matrix(const Json &config);

        ~Matrix();

        /**
         * change shape of the matrix
         * @param dimensions  stored in an vector
         */
        void resize(const vector<int> &dimensions);

        /**
         * allolcate a block of memory initialized with 0.0
         */
        void reallocate(float value = 0.0);

        /**
         * get dimension size
         * @param index
         * @return
         */
        inline int dimension(int index) const {
            if (index < 0) {
                index += _dimensions_size;
            }
            if (index < 0 || index >= _dimensions_size) {
                throw_exception("dimension index is out of bounds");
            }
            return _dimensions[index];
        }

        /**
         * count data size
         * @param begin
         * @return
         */
        inline int count(int begin = 0) const {
            return count(begin, _dimensions_size);
        }

        /**
         * count data size
         * @param begin
         * @param end
         * @return
         */
        inline int count(int begin, int end) const {
            return std::accumulate(_dimensions.begin() + begin, _dimensions.begin() + end, 1,
                                   [](int a, int b) { return a * b; });
        }

        /**
         * get the data ptr
         * @return
         */
        inline float *get_data() {
            return _data;
        }

        /**
         * set data
         * @param data
         */
        inline void set_data(float *data) {
            clear_data();
            _data = data;
            _is_external_memory = true;
        }

        inline bool *get_sign_data() {
            return sign_data;
        }

        /**
         * set data
         * @param data
         */
        inline void set_data(bool *data) {
            clear_data();
            sign_data = data;
            _is_external_memory = true;
        }

        inline string get_name() {
            return _name;
        }

        inline void set_name(string name) {
            _name = name;
        }

        inline float at(const vector<int> &indices) {
            return _data[offset(indices)];
        }

        /**
         * caculate the offset
         * @param indices
         * @return
         */
        inline int offset(const vector<int> &indices) const {
            int i, offset = 0, indices_size = indices.size(), common_size = std::min(_dimensions_size, indices_size);
            for (i = 0; i < common_size; i++) {
                offset = offset * _dimensions[i] + indices[i];
            }
            for (; i < _dimensions_size; i++) {
                offset = offset * _dimensions[i];
            }
            return offset;
        }

        /**
         * get the dementions
         * @return
         */
        inline vector<int> get_dimensions() const {
            return _dimensions;
        }

        /**
         * check data for debug
         * @return
         */
        string descript() {
            auto data = get_data();
            int k = 15;
            int c = count() / k;
            stringstream ss;
            if (count() < k) {
                k = count();
                c = 1;
            }
            for (int i = 0; i < k; i++) {
                ss << data[i * c] << " ";
            }


            return ss.str();
        }

        /**
             * check data for debug
             * @return
             */
        string descript_dimention() {

            stringstream ss;
            for (int i = 0; i < _dimensions.size(); i++) {
                ss << _dimensions[i] << " ";
            }
            return ss.str();
        }

        void dump(string file_name) {
            std::ofstream datafile(file_name.c_str());
            if (!datafile.is_open()) {
                cout << file_name << " open failed" << endl;
            }
            for (int j = 0; j < count(); ++j) {
                datafile << get_data()[j] << " ";
            }
            datafile.close();
        }

    private:
        string _name;
        // original data
        float *_data;

        bool *sign_data;

        vector<int> _dimensions;

        int _data_size;

        int _dimensions_size;

        bool _is_external_memory;

        void clear_data();
    };
};

#endif
