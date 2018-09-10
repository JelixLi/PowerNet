

#include "commons/commons.h"

namespace mdl {
    const char *log_tag = "MDL LOG built on " __DATE__ " " __TIME__;

    const int string_size = 30;

    const int model_version = 1;

    const string matrix_name_data = "data";

    const string matrix_name_test_data = "test-data";

    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }
    //
    // void im2col(const float *data_im, const int channels, const int height,
    //             const int width, const int kernel_size,
    //             const int pad, const int stride, float *data_col) {
    //     const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    //     const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
    //     const int channel_size = height * width;
    //     for (int channel = channels; channel--; data_im += channel_size) {
    //         for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
    //             for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
    //                 int input_row = -pad + kernel_row;
    //                 for (int output_rows = output_h; output_rows; output_rows--) {
    //                     if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
    //                         for (int output_cols = output_w; output_cols; output_cols--) {
    //                             *(data_col++) = 0;
    //                         }
    //                     } else {
    //                         int input_col = -pad + kernel_col;
    //                         for (int output_col = output_w; output_col; output_col--) {
    //                             if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
    //                                 *(data_col++) = data_im[input_row * width + input_col];
    //                             } else {
    //                                 *(data_col++) = 0;
    //                             }
    //                             input_col += stride;
    //                         }
    //                     }
    //                     input_row += stride;
    //                 }
    //             }
    //         }
    //     }
    // }


    void im2col(const float *data_im, const int channels, const int height,
                const int width, const int kernel_size,
                const int pad, const int stride, float *data_col) {
        const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
        const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
        const int channel_size = height * width;
        const int col_size=output_h*output_w*kernel_size*kernel_size;


        const float *bak_img_ptr_chan_1=data_im;
        const float *bak_img_ptr_chan_2=data_im+channel_size;
        const float *bak_img_ptr_chan_3=data_im+2*channel_size;
        const float *bak_img_ptr_chan_4=data_im+3*channel_size;

        register const float *ptr_img_chan_1;
        register const float *ptr_img_chan_2;
        register const float *ptr_img_chan_3;
        register const float *ptr_img_chan_4;


        float *bak_col_ptr_chan_1=data_col;
        float *bak_col_ptr_chan_2=data_col+col_size;
        float *bak_col_ptr_chan_3=data_col+2*col_size;
        float *bak_col_ptr_chan_4=data_col+3*col_size;

        register float *ptr_col_chan_1;
        register float *ptr_col_chan_2;
        register float *ptr_col_chan_3;
        register float *ptr_col_chan_4;

        int _c=channels/4;
        int _k=channels%4;

        int _u=output_w/4;
        int _v=output_w%4;

        int offset;

        for(int inc=0;inc<_c;inc++) {
            ptr_col_chan_1=bak_col_ptr_chan_1+inc*col_size*4;
            ptr_col_chan_2=bak_col_ptr_chan_2+inc*col_size*4;
            ptr_col_chan_3=bak_col_ptr_chan_3+inc*col_size*4;
            ptr_col_chan_4=bak_col_ptr_chan_4+inc*col_size*4;

            ptr_img_chan_1=bak_img_ptr_chan_1+inc*channel_size*4;
            ptr_img_chan_2=bak_img_ptr_chan_2+inc*channel_size*4;
            ptr_img_chan_3=bak_img_ptr_chan_3+inc*channel_size*4;
            ptr_img_chan_4=bak_img_ptr_chan_4+inc*channel_size*4;

            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    int input_row = -pad + kernel_row;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            // for (int output_cols = output_w; output_cols; output_cols--) {
                            //     *(data_col++) = 0;
                            // }
                            for (int output_cols = 0; output_cols<_u; output_cols++) {
                                *(ptr_col_chan_1)=0;
                                *(ptr_col_chan_1+1)=0;
                                *(ptr_col_chan_1+2)=0;
                                *(ptr_col_chan_1+3)=0;

                                *(ptr_col_chan_2)=0;
                                *(ptr_col_chan_2+1)=0;
                                *(ptr_col_chan_2+2)=0;
                                *(ptr_col_chan_2+3)=0;

                                *(ptr_col_chan_3)=0;
                                *(ptr_col_chan_3+1)=0;
                                *(ptr_col_chan_3+2)=0;
                                *(ptr_col_chan_3+3)=0;

                                *(ptr_col_chan_4)=0;
                                *(ptr_col_chan_4+1)=0;
                                *(ptr_col_chan_4+2)=0;
                                *(ptr_col_chan_4+3)=0;

                                ptr_col_chan_1+=4;
                                ptr_col_chan_2+=4;
                                ptr_col_chan_3+=4;
                                ptr_col_chan_4+=4;
                            }

                            for(int output_cols=0;output_cols<_v;output_cols++) {
                                *(ptr_col_chan_1++)=0;
                                *(ptr_col_chan_2++)=0;
                                *(ptr_col_chan_3++)=0;
                                *(ptr_col_chan_4++)=0;
                            }

                        } else {
                            int input_col = -pad + kernel_col;
                            // for (int output_col = output_w; output_col; output_col--) {
                            //     if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                            //         *(data_col++) = data_im[input_row * width + input_col];
                            //     } else {
                            //         *(data_col++) = 0;
                            //     }
                            //     input_col += stride;
                            //
                            offset=input_row*width+input_col;
                            for(int output_cols=0;output_cols<_u;output_cols++) {

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1)=ptr_img_chan_1[offset];
                                    *(ptr_col_chan_2)=ptr_img_chan_2[offset];
                                    *(ptr_col_chan_3)=ptr_img_chan_3[offset];
                                    *(ptr_col_chan_4)=ptr_img_chan_4[offset];

                                } else {
                                    *(ptr_col_chan_1)=0;
                                    *(ptr_col_chan_2)=0;
                                    *(ptr_col_chan_3)=0;
                                    *(ptr_col_chan_4)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+1)=ptr_img_chan_1[offset];
                                    *(ptr_col_chan_2+1)=ptr_img_chan_2[offset];
                                    *(ptr_col_chan_3+1)=ptr_img_chan_3[offset];
                                    *(ptr_col_chan_4+1)=ptr_img_chan_4[offset];

                                } else {
                                    *(ptr_col_chan_1+1)=0;
                                    *(ptr_col_chan_2+1)=0;
                                    *(ptr_col_chan_3+1)=0;
                                    *(ptr_col_chan_4+1)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+2)=ptr_img_chan_1[offset];
                                    *(ptr_col_chan_2+2)=ptr_img_chan_2[offset];
                                    *(ptr_col_chan_3+2)=ptr_img_chan_3[offset];
                                    *(ptr_col_chan_4+2)=ptr_img_chan_4[offset];

                                } else {
                                    *(ptr_col_chan_1+2)=0;
                                    *(ptr_col_chan_2+2)=0;
                                    *(ptr_col_chan_3+2)=0;
                                    *(ptr_col_chan_4+2)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+3)=ptr_img_chan_1[offset];
                                    *(ptr_col_chan_2+3)=ptr_img_chan_2[offset];
                                    *(ptr_col_chan_3+3)=ptr_img_chan_3[offset];
                                    *(ptr_col_chan_4+3)=ptr_img_chan_4[offset];

                                } else {
                                    *(ptr_col_chan_1+3)=0;
                                    *(ptr_col_chan_2+3)=0;
                                    *(ptr_col_chan_3+3)=0;
                                    *(ptr_col_chan_4+3)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                ptr_col_chan_1+=4;
                                ptr_col_chan_2+=4;
                                ptr_col_chan_3+=4;
                                ptr_col_chan_4+=4;

                            }


                            for(int output_cols=0;output_cols<_v;output_cols++) {

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1++)=ptr_img_chan_1[offset];
                                    *(ptr_col_chan_2++)=ptr_img_chan_2[offset];
                                    *(ptr_col_chan_3++)=ptr_img_chan_3[offset];
                                    *(ptr_col_chan_4++)=ptr_img_chan_4[offset];

                                } else {
                                    *(ptr_col_chan_1++)=0;
                                    *(ptr_col_chan_2++)=0;
                                    *(ptr_col_chan_3++)=0;
                                    *(ptr_col_chan_4++)=0;
                                }

                                input_col+=stride;
                                offset+=stride;
                            }
                        }
                        input_row += stride;
                    }
                }
            }


        }


        for(int inc=0;inc<_k;inc++) {
            ptr_col_chan_1=bak_col_ptr_chan_1+inc*col_size+_c*4*col_size;
            ptr_img_chan_1=bak_img_ptr_chan_1+inc*channel_size+_c*4*channel_size;

            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                    int input_row = -pad + kernel_row;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            // for (int output_cols = output_w; output_cols; output_cols--) {
                            //     *(data_col++) = 0;
                            // }
                            for (int output_cols = 0; output_cols<_u; output_cols++) {
                                *(ptr_col_chan_1)=0;
                                *(ptr_col_chan_1+1)=0;
                                *(ptr_col_chan_1+2)=0;
                                *(ptr_col_chan_1+3)=0;

                                ptr_col_chan_1+=4;
                            }

                            for(int output_cols=0;output_cols<_v;output_cols++) {
                                *(ptr_col_chan_1++)=0;
                            }

                        } else {
                            int input_col = -pad + kernel_col;
                            // for (int output_col = output_w; output_col; output_col--) {
                            //     if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                            //         *(data_col++) = data_im[input_row * width + input_col];
                            //     } else {
                            //         *(data_col++) = 0;
                            //     }
                            //     input_col += stride;
                            //
                            offset=input_row*width+input_col;
                            for(int output_cols=0;output_cols<_u;output_cols++) {

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1)=ptr_img_chan_1[offset];

                                } else {
                                    *(ptr_col_chan_1)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+1)=ptr_img_chan_1[offset];
                                } else {
                                    *(ptr_col_chan_1+1)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+2)=ptr_img_chan_1[offset];

                                } else {
                                    *(ptr_col_chan_1+2)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1+3)=ptr_img_chan_1[offset];
                                } else {
                                    *(ptr_col_chan_1+3)=0;
                                }
                                input_col+=stride;
                                offset+=stride;

                                ptr_col_chan_1+=4;

                            }


                            for(int output_cols=0;output_cols<_v;output_cols++) {

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(ptr_col_chan_1++)=ptr_img_chan_1[offset];

                                } else {
                                    *(ptr_col_chan_1++)=0;
                                }

                                input_col+=stride;
                                offset+=stride;
                            }
                        }
                        input_row += stride;
                    }
                }
            }
        }

    }



    Time time() {
        return std::chrono::high_resolution_clock::now();
    }

    double time_diff(Time t1, Time t2) {
        typedef std::chrono::microseconds ms;
        auto diff = t2 - t1;
        ms counter = std::chrono::duration_cast<ms>(diff);
        return counter.count() / 1000.0;
    }

    void idle(const char *fmt, ...) {
    }

    bool equal(float a, float b) {
        const float EPSILON = 1e-5;
        if (fabsf(a - b) < EPSILON) {
            return true;
        }
        return false;

    }

    void copy(int length, float *x, float *y) {
        if (x != y) {
            memcpy(y, x, sizeof(float) * length);
        }
    }
};
