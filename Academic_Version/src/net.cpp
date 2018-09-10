
#include "net.h"
#include "base/layer.h"
#include "base/matrix.h"
#include "loader/loader.h"

#include "layer/concat_layer.h"
#include "layer/convolution_layer.h"
#include "layer/fc_layer.h"
#include "layer/pooling_layer.h"
#include "layer/relu_layer.h"
#include "layer/split_layer.h"
#include "layer/lrn_layer.h"
#include "layer/batch_normal_layer.h"
#include "layer/scale_layer.h"
#include "layer/soft_max_layer.h"
#include "layer/sigmoid_layer.h"
#include "layer/elt_wise_layer.h"

#include <thread>

#ifdef NEED_DUMP
#include <limits>
#include <cstdio>
#endif

namespace mdl {


    Net::Net(const Json &config) : _thread_num(1) {
        for (const auto &layer_config: config["layer"].array_items()) {
            Layer *layer = nullptr;
            string type = layer_config["type"].string_value();

            if (type == "ConcatLayer") {
                layer = new ConcatLayer(layer_config);
            } else if (type == "ConvolutionLayer") {
                ConvolutionLayer *conv_layer= new ConvolutionLayer(layer_config);
                layer = conv_layer;
                convptr.push_back(conv_layer);
            } else if (type == "FCLayer") {
                layer = new FCLayer(layer_config);
            } else if (type == "PoolingLayer") {
                layer = new PoolingLayer(layer_config);
            } else if (type == "ReluLayer") {
                layer = new ReluLayer(layer_config);
            } else if (type == "SplitLayer") {
                layer = new SplitLayer(layer_config);
            } else if (type == "LrnLayer") {
                layer = new LrnLayer(layer_config);
            } else if (type == "ScaleLayer") {
                layer = new ScaleLayer(layer_config);
            } else if (type == "BatchNormLayer") {
                layer = new BatchNormalLayer(layer_config);
            } else if (type == "SoftmaxLayer") {
                layer = new SoftmaxLayer(layer_config);
            } else if (type == "SigmoidLayer") {
                layer = new SigmoidLayer(layer_config);
            } else if (type == "EltwiseLayer") {
                layer = new EltWiseLayer(layer_config);
            }

            if (layer) {
                _layers.push_back(layer);
            } else {
                throw_exception("could not create [%s] layer", layer_config["name"].string_value().c_str());
            }
        }

    }

    Net::~Net() {
        for (auto &layer: _layers) {
            delete layer;
        }
        _layers.clear();
    }

    void forward_from_to_async(int pid, int inception_thread_num, int start, int end, vector<Layer *> layers) {
        for (int i = start; i < end; i++) {
            auto type = layers[i]->layer_type();
            if (type == LayerType::CONCAT) {
                return;
            }
            int layer_pid = layers[i]->pid();
            if ((layer_pid % inception_thread_num + 1) != pid) {
                continue;
            }
            layers[i]->forward();
        }
    }



    vector<float> Net::forward_from_to(float *image, int start, int end, bool sampling) {
        vector<float> result;
        if (image == nullptr) {
            Matrix *test_data_matrix = Loader::shared_instance()->_matrices[matrix_name_test_data];
            if (test_data_matrix == nullptr) {
                throw_exception("test_data_matrix is nullptr");
            }
            image = test_data_matrix->get_data();

        }
        if (image == nullptr) {
            throw_exception("image is still nullptr");
        }


        Matrix *data_matrix = Loader::shared_instance()->_matrices[matrix_name_data];
        if (data_matrix == nullptr) {
            throw_exception("data_matrix is nullptr");
        }

        data_matrix->set_data(image);

        if (end > _layers.size()) {
            throw_exception("the end layer %d is out of bounds", end);
        }

#ifdef MULTI_THREAD
        bool threads_enable = false;
        bool in_threads = false;
        int inception_count = 0;
        for (int i = start; i < end; i++) {
            auto type = _layers[i]->layer_type();
            if (threads_enable && inception_count < MAX_INCEPTION_NUM) {
                std::thread *ths = new std::thread[_thread_num];
                for (int j = 1; j <= _thread_num; j++) {
                    ths[j - 1] = std::thread(forward_from_to_async, j, _thread_num, i, end, _layers);
                }
                for (int j = 0; j < _thread_num; ++j) {
                    ths[j].join();

                }
                delete []ths;
                threads_enable = false;
                in_threads = true;
                continue;
            }
            if (type == LayerType::SPLIT && _thread_num > 1) {
                threads_enable = true;
            }
            if (in_threads && inception_count < MAX_INCEPTION_NUM) {
                if (type != LayerType::CONCAT) {
                    continue;
                } else {
                    in_threads = false;
                    inception_count++;
                }
            }
            int thread_num = _thread_num;
            _layers[i]->forward(thread_num);
        }
#else

        for (int i = 0; i < _layers.size(); i++) {

           //cout << _layers[i]->name() << " input ="<< _layers[i]->input()[0]->descript()<<endl;
            _layers[i]->forward();
           //cout << _layers[i]->name() << "  output = "<<_layers[i]->output()[0]->descript()<<endl;
        }


#endif

        Matrix *matrix = _layers[end - 1]->output().front();
        float *matrix_data = matrix->get_data();
        int count = matrix->count();
        int sample_count = 10;
        if (sampling && count > sample_count) {
            for (int i = 0; i < count; i += count / sample_count) {
                result.push_back(matrix_data[i]);
            }
        } else {
            for (int i = 0; i < count; i++) {
                result.push_back(matrix_data[i]);
            }
        }
        return result;
    }

    vector<float> Net::predict(float *image) {
        int start = 0;
        int end = _layers.size();

        vector<float> temp=Net::forward_from_to(image, start, end);
        return temp;
    }

    // void DebugPrint(vector<Matrix*> item)
    // {
    //     for(vector<Matrix*>::iterator cit=item.begin();
    //     cit!=item.end();
    //     ++cit)
    //     {
    //         Matrix *mp=*cit;
    //         cout<<"name: "<<mp->get_name()<<" ";
    //         vector<int> dim=mp->get_dimensions();
    //         cout<<"dimensions: ";
    //         for(int i=0;i<dim.size();i++)
    //         {
    //             cout<<dim[i];
    //             if(i!=dim.size()-1)
    //                 cout<<"X";
    //         }
    //         cout<<endl;
    //         int data_size=mp->count();
    //         float *data=mp->get_data();
    //         for(int j=0;j<data_size;j++)
    //         {
    //             cout<<data[j]<<" ";
    //             if((j+1)%5==0)
    //                 cout<<"\n";
    //         }
    //         cout<<"\n"<<endl;
    //     }
    // }
    //

    unsigned int getBitWeight(int s,float a)
    {
        // int delta=0;
        // if(a>delta)
        //     return 2u<<s;
        // else if(a<delta)
        //     return 3u<<s;
        // else
        //     return 0u;
        if(a>0)
            return 1;
        else
            return 0;
    }

    void Bit_Transform(int m,int k,float *orig_data)
    {
        int BK=k/16;
        int _BK=k%16;
        if(_BK>0)
            BK++;

        int bit_pos=0;
        unsigned int *bit_buffer=new unsigned int[m*BK];
        memset(bit_buffer,0,sizeof(bit_buffer));
        float *sp=orig_data;
        unsigned int *bs=bit_buffer;
        for(int j=0;j<m;j++)
        {
            for(int i=0;i<k;i++)
            {
                if(bit_pos==32)
                {
                    bs++;
                    bit_pos=0;
                }
                (*bs)^=getBitWeight(bit_pos,*(sp++));
                bit_pos+=2;
            }
            bs++;
            bit_pos=0;
        }
        memcpy((unsigned int*)orig_data,bit_buffer,m*BK*sizeof(unsigned int));
        delete [] bit_buffer;
    }



    void Net::Transform_Conv()
    {
        int n=convptr.size();
        for(int i=0;i<n;i++)
        {
            if(convptr[i]->get_kernel_size()!=1)
            {
                Layer *p=convptr[i];

                //  DebugPrint(p->output());
                //DebugPrint(p->weight());

                vector<Matrix*> weight=p->weight();
                for(int j=0;j<weight.size();j++)
                {
                    int k=weight[j]->count(1);   // input_channels * kernel_h * kernel_w
                    float *orig_data=weight[j]->get_data();
                    int m=weight[j]->dimension(0);

                    Bit_Transform(m,k,orig_data);
                }

            }

        }
    }



#ifdef NEED_DUMP
    void Net::dump(string filename) {
        dump_with_quantification(filename);
    }
    /**
     *
     * dump out the params with quantification
     * @param filename
     */
    void Net::dump_with_quantification(string filename) {
        int total_size = 0;
        int matrix_count = 0;
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                matrix_count++;
                total_size += matrix->count() * sizeof(uint8_t);
            }
        }
        Matrix *test_data_matrix = Loader::shared_instance()->_matrices[matrix_name_test_data];
        if (test_data_matrix) {
            matrix_count++;
            total_size += test_data_matrix->count() * sizeof(uint8_t);
        }
        total_size += matrix_count * (string_size * sizeof(char) + sizeof(int) + 2 * sizeof(float));
        total_size += 3 * sizeof(int);

        FILE *out_file = fopen(filename.c_str(), "wb");
        fwrite(&total_size, sizeof(int), 1, out_file);
        fwrite(&model_version, sizeof(int), 1, out_file);
        fwrite(&matrix_count, sizeof(int), 1, out_file);

        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                int matrix_size = matrix->count();
                fwrite(&matrix_size, sizeof(int), 1, out_file);
            }
        }
        if (test_data_matrix) {
            int matrix_size = test_data_matrix->count();
            fwrite(&matrix_size, sizeof(int), 1, out_file);
        }

        vector<float> min_values;
        vector<float> max_values;
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                float min_value = std::numeric_limits<float>::max();
                float max_value = std::numeric_limits<float>::min();
                for (int i = 0; i < matrix->count(); i++) {
                    min_value = min(min_value, matrix->get_data()[i]);
                    max_value = max(max_value, matrix->get_data()[i]);
                }
                min_values.push_back(min_value);
                max_values.push_back(max_value);
                fwrite(&min_value, sizeof(float), 1, out_file);
                fwrite(&max_value, sizeof(float), 1, out_file);
            }
        }
        if (test_data_matrix) {
            float min_value = std::numeric_limits<float>::max();
            float max_value = std::numeric_limits<float>::min();
            for (int i = 0; i < test_data_matrix->count(); i++) {
                min_value = min(min_value, test_data_matrix->get_data()[i]);
                max_value = max(max_value, test_data_matrix->get_data()[i]);
            }
            min_values.push_back(min_value);
            max_values.push_back(max_value);
            fwrite(&min_value, sizeof(float), 1, out_file);
            fwrite(&max_value, sizeof(float), 1, out_file);
        }

        char matrix_name[string_size];
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                strcpy(matrix_name, matrix->get_name().c_str());
                fwrite(matrix_name, sizeof(char), string_size, out_file);
            }
        }
        if (test_data_matrix) {
            strcpy(matrix_name, matrix_name_data.c_str());
            fwrite(matrix_name, sizeof(char), string_size, out_file);
        }

        int matrix_index = 0;
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                float min_value = min_values[matrix_index];
                float max_value = max_values[matrix_index];
                for (int i = 0; i < matrix->count(); i++) {
                    float value = matrix->get_data()[i];
                    uint8_t factor = (uint8_t)round((value - min_value) / (max_value - min_value) * 255);
                    fwrite(&factor, sizeof(uint8_t), 1, out_file);
                }
                matrix_index++;
            }
        }
        if (test_data_matrix) {
            float min_value = min_values[matrix_index];
            float max_value = max_values[matrix_index];
            for (int i = 0; i < test_data_matrix->count(); i++) {
                float value = test_data_matrix->get_data()[i];
                uint8_t factor = (uint8_t)round((value - min_value) / (max_value - min_value) * 255);
                fwrite(&factor, sizeof(uint8_t), 1, out_file);
            }
            matrix_index++;
        }

        fclose(out_file);
    }
    /**
     * dump out the params without quantification
     * @param filename
     */
    void Net::dump_without_quantification(string filename) {
        int total_size = 0;
        int matrix_count = 0;
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                matrix_count++;
                total_size += matrix->count() * sizeof(float);
            }
        }
        Matrix *test_data_matrix = Loader::shared_instance()->_matrices[matrix_name_test_data];
        if (test_data_matrix) {
            matrix_count++;
            total_size += test_data_matrix->count() * sizeof(float);
        }
        total_size += matrix_count * (string_size + sizeof(int));
        total_size += 3 * sizeof(int);

        FILE *out_file = fopen(filename.c_str(), "wb");
        fwrite(&total_size, sizeof(int), 1, out_file);
        fwrite(&model_version, sizeof(int), 1, out_file);
        fwrite(&matrix_count, sizeof(int), 1, out_file);

        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                int matrix_size = matrix->count();
                fwrite(&matrix_size, sizeof(int), 1, out_file);
            }
        }
        if (test_data_matrix) {
            int matrix_size = test_data_matrix->count();
            fwrite(&matrix_size, sizeof(int), 1, out_file);
        }

        char matrix_name[string_size];
        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                strcpy(matrix_name, matrix->get_name().c_str());
                fwrite(matrix_name, sizeof(char), string_size, out_file);
            }
        }
        if (test_data_matrix) {
            strcpy(matrix_name, matrix_name_data.c_str());
            fwrite(matrix_name, sizeof(char), string_size, out_file);
        }

        for (auto &layer: _layers) {
            for (auto &matrix: layer->weight()) {
                fwrite(matrix->get_data(), sizeof(float), matrix->count(), out_file);
            }
        }
        if (test_data_matrix) {
            fwrite(test_data_matrix->get_data(), sizeof(float), test_data_matrix->count(), out_file);
        }

        fclose(out_file);
    }
#endif
};
