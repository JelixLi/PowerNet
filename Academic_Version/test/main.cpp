
#include <iostream>
#include <dirent.h>
#include <cstdio>

#include "net.h"
#include "base/matrix.h"
#include "loader/loader.h"
#include "math/gemm.h"

using namespace std;

int run();

float *getImage(string path)
{
    FILE *file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw_exception("can't open image file");
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    if (size <= 0 || size < sizeof(int)) {
        throw_exception("binary file size is too small");
    }

    rewind(file);
    
    char *data = new char[size];
    size_t bytes_read = fread(data, 1, size, file);
    if (bytes_read != size) {
        throw_exception("read binary file bytes do not match with fseek");
    }
    fclose(file);

    return (float*)data;
}

int main() {

    int run_count = 1;
    for (int i = 0; i < run_count; i++) {
        cout << "start running cycle : " << i << endl;
//        EXCEPTION_HEADER
        run();
//        EXCEPTION_FOOTER
        cout << "end running cycle : " << i << endl;
    }
}

bool is_equal(float a, float b) {
    // epsilon is too strict about correctness
    // return abs(a - b) <= std::numeric_limits<float>::epsilon();
    return abs(a - b) <= 0.001;
}

bool is_correct_result(vector<float> &result) {
    // // the correct result without quantification is : 87.5398 103.573 209.723 196.812
    // vector<float> correct_result{87.5398, 103.573, 209.723, 196.812};
    // the correct result with quantification is : 87.4985 103.567 209.752 196.71
    vector<float> correct_result{64.777, 101.88, 210.735, 199.144};
    if (result.size() != 4) {
        return false;
    }
    for (int i = 0; i < 4; i++) {
        if (!is_equal(result[i], correct_result[i])) {
            return false;
        }
    }
    return true;
}

int find_max(vector<float> data) {
    float max = -1000000;
    int index = 0;
    for (int i = 0; i < data.size(); ++i) {
        if (data[i] > max) {
            max = data[i];
            index = i + 1;
        }
    }
    return index;
}

int run() {
    // thread num should set 1 while using mobilenet & resnet
    int thread_num = 3;
    if (mdl::Gemmer::gemmers.size() == 0) {
        for (int i = 0; i < max(thread_num, 3); i++) {
            mdl::Gemmer::gemmers.push_back(new mdl::Gemmer());
        }
    }
    mdl::Loader *loader = mdl::Loader::shared_instance();

    std::string prefix("./model/googlenet/");
    auto t1 = mdl::time();
    bool load_success = loader->load(prefix + "g_model.min.json", prefix + "g_data.min.bin");

    // std::string prefix("./model/mobilenet/");
    // auto t1 = mdl::time();
    // bool load_success = loader->load(prefix + "m_model.min.json", prefix + "m_data.min.bin");

    if (!load_success) {
        cout << "load failure" << endl;
        loader->clear();
        return -1;
    }
    if (!loader->get_loaded()) {
        throw_exception("loader is not loaded yet");
    }
    mdl::Net *net = new mdl::Net(loader->_model);
    net->set_thread_num(thread_num);

#ifdef OPTIMIZE
    net->Transform_Conv();
#endif

    auto t2 = mdl::time();
    cout << "load time : " << mdl::time_diff(t1, t2) << "ms" << endl;

    int count = 1;
    double total = 0;
    vector<float> result;
    for (int i = 0; i < count; i++) {
        Time t1 = mdl::time();
        result = net->predict(nullptr);
        //result = net->predict(getImage(string("/home/jieli/code/mobile-deep-learning/test/images/data.dat")));
        Time t2 = mdl::time();
        double diff = mdl::time_diff(t1, t2);
        total += diff;
    }
    cout << "total cost: " << total / count << "ms." << endl;
    for (float num: result) {
        cout << num << " ";
    }
    cout <<endl;
    // uncomment while testing clacissification models
//    cout << "the max prob index = "<<find_max(result)<<endl;
    cout << "Done!" << endl;
//    cout << "it " << (is_correct_result(result) ? "is" : "isn't") << " a correct result." << endl;
    loader->clear();
    delete net;
    return 0;
}
