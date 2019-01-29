#ifndef GPU_GEMM_H
#define GPU_GEMM_H

#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>
#include <math.h>

#ifdef GPU

#include "QPULib.h"

namespace Power {

  typedef Kernel<Ptr<Float>, Ptr<Float>, Ptr<Float>,Int,Int,Int> GemmKernelType;

  template<typename T>
  class GManager {
  public:
  	GManager();

    void LoadDataIntoGpu(
      SharedArray<T> &_shared_array_buffer,
      T *input_data_buffer,
      int group_size,
      int data_size);

    void LoadColDataIntoGpu(
      SharedArray<T> &_shared_array_buffer,
      T *input_data_buffer,
      int step_size,
      int group_size,
      int data_size);

    void TransInput2GpuFormat(
      T *input_data_buffer,
      const T *input_data,
      int input_height,
      int input_width,
      int input_channel,
      int kernel_size,
      int pad,
      int stride);

    void GetOutputFromGpu(
      SharedArray<T> &_shared_array_buffer,
      T *output_data_buffer,
      int offset,
      int step_size,
      int row_size,
      int col_size);

    void gpu_conv(
      T *weight,
      T *input,
      T *output,
      int height,
      int width,
      int channels,
      int kernel_size,
      int output_num,
      int pad,
      int stride);


    // void TransBiasToGpuFormat(float *bias_buffer,const float *bias_data,int m,int n);

    void set_GemmKernel(GemmKernelType *GemmKernel) {
      this->GemmKernel = GemmKernel;
      this->GemmKernel->setNumQPUs(12);
    }

  private:
  	void Init_Gpu_Memory();


  	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  	    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
  	}

    GemmKernelType *GemmKernel;

  	SharedArray<T> _gp_array[3];

    int Max_GPU_Memory; // float(4 bytes)

    int Gpu_Memory_Basic_Block;
  };


  // template<typename T>
  // void TransBiasToGpuFormat(float *bias_buffer,const float *bias_data,int m,int n) {
  //     int m_group = m / Gpu_Memory_Basic_Block;
  //     int _m_group = m % Gpu_Memory_Basic_Block;

  //     int n_group = n / Gpu_Memory_Basic_Block;
  //     int _n_group = n % Gpu_Memory_Basic_Block;

  //     register T *bias_buffer_ptr = bias_buffer;

  //     for(int i=0;i<m_group+1;i++) {
  //       int weight_offset = i*k*Gpu_Memory_Basic_Block;
  //       int weight_group_size = ((i==m_group||m_group==0)?_m_group:Gpu_Memory_Basic_Block);

  //       for(int j=0;j<n_group+1;j++) {
  //         int input_offset = j*Gpu_Memory_Basic_Block;
  //         int input_group_size = ((j==n_group||n_group==0)?_n_group:Gpu_Memory_Basic_Block);

  //         int offset = i*Gpu_Memory_Basic_Block*n+j*Gpu_Memory_Basic_Block;
  //         int step_size = n;
  //         int row_size = weight_group_size;
  //         int col_size = input_group_size;

  //         const register T *data_buffer_ptr;
  //         int row_offset = offset / step_size;
  //         int col_offset = offset % step_size;

  //         int n = col_size / 4;
  //         int _n = col_size % 4;
  //         register int base_addr;
  //         for(int i=0;i<row_size;i++) { 
  //           base_addr = (i+row_offset)*step_size+col_offset;
  //           data_buffer_ptr = bias_data + base_addr;
  //           for(register int j=0;j<n;j+=4) {
  //             *bias_buffer_ptr++ = *data_buffer_ptr++;
  //             *bias_buffer_ptr++ = *data_buffer_ptr++;
  //             *bias_buffer_ptr++ = *data_buffer_ptr++;
  //             *bias_buffer_ptr++ = *data_buffer_ptr++;
  //           }

  //           for(register int j=0;j<_n;j++) {
  //             *bias_buffer_ptr++ = *data_buffer_ptr++;
  //           }

  //         }

  //     }
  //   }
  // }

  template<typename T>
  void GManager<T>::gpu_conv(
    T *weight,
    T *input,
    T *output,
    int height,
    int width,
    int channels,
    int kernel_size,
    int output_num,
    int pad,
    int stride) {

      if(GemmKernel==NULL) {
        std::cout<<"please init Kernel"<<std::endl;
        return;
      }

      int output_h = (height + 2 * pad - kernel_size) / stride + 1;
      int output_w = (width + 2 * pad - kernel_size) / stride + 1;

      int row_padding = 16 - (channels*kernel_size*kernel_size) % 16;

      int row_size = channels*kernel_size*kernel_size + row_padding;
      int col_size = output_h*output_w;

      int m = output_num;
      int n = col_size;
      int k = row_size;

      SharedArray<T>& weight_buffer = _gp_array[0];
      SharedArray<T>& input_buffer = _gp_array[1];
      SharedArray<T>& output_buffer = _gp_array[2];

      int m_group = m / Gpu_Memory_Basic_Block;
      int _m_group = m % Gpu_Memory_Basic_Block;

      int n_group = n / Gpu_Memory_Basic_Block;
      int _n_group = n % Gpu_Memory_Basic_Block;


      for(int i=0;i<m_group+1;i++) {
        int weight_offset = i*k*Gpu_Memory_Basic_Block;
        int weight_group_size = ((i==m_group||m_group==0)?_m_group:Gpu_Memory_Basic_Block);

        LoadDataIntoGpu(
          weight_buffer,
          weight+weight_offset,
          weight_group_size,
          k);

        for(int j=0;j<n_group+1;j++) {
          // int input_offset = j*k*Gpu_Memory_Basic_Block;
          int input_offset = j*Gpu_Memory_Basic_Block;
          int input_group_size = ((j==n_group||n_group==0)?_n_group:Gpu_Memory_Basic_Block);

          // LoadDataIntoGpu(
          //   input_buffer,
          //   input+input_offset,
          //   input_group_size,
          //   k);

          LoadColDataIntoGpu(
            input_buffer,
            input+input_offset,
            n,
            input_group_size,
            k);

          (*GemmKernel)(
            &weight_buffer,
            &input_buffer,
            &output_buffer,
            weight_group_size,
            input_group_size,
            k);

          GetOutputFromGpu(
            output_buffer,
            output,
            i*Gpu_Memory_Basic_Block*n+j*Gpu_Memory_Basic_Block,
            n,
            weight_group_size,
            input_group_size);
      }
    }
  }

  template<typename T>
  void GManager<T>::GetOutputFromGpu(
      SharedArray<T> &_shared_array_buffer,
      T *output_data_buffer,
      int offset,
      int step_size,
      int row_size,
      int col_size) {

      register T *output_data_buffer_ptr;
      int row_offset = offset / step_size;
      int col_offset = offset % step_size;
      register int pos = 0;

      int n = col_size / 4;
      int _n = col_size % 4;
      register int base_addr;
      for(int i=0;i<row_size;i++) { 
        base_addr = (i+row_offset)*step_size+col_offset;
        output_data_buffer_ptr = output_data_buffer + base_addr;
        for(register int j=0;j<n;j+=4) {
          *output_data_buffer_ptr++ = _shared_array_buffer[pos];
          *output_data_buffer_ptr++ = _shared_array_buffer[pos+16];
          *output_data_buffer_ptr++ = _shared_array_buffer[pos+32];
          *output_data_buffer_ptr++ = _shared_array_buffer[pos+48];
          pos += 64;
        }

        for(register int j=0;j<_n;j++) {
          *output_data_buffer_ptr++ = _shared_array_buffer[pos];
          pos += 16;
        }

      }

  }

  template<typename T>
  void GManager<T>::TransInput2GpuFormat(
    T *output_data,
    const T *input_data_buffer,
    int input_height,
    int input_width,
    int input_channel,
    int kernel_size,
    int pad,
    int stride) {

      const T *input_data;

      for(int row=-pad;row<input_height+pad-kernel_size+1;row+=stride) {
          for(int col=-pad;col<input_width+pad-kernel_size+1;col+=stride) {

              for(int chan=0;chan<input_channel;chan++) {

                  input_data = input_data_buffer + chan*input_height*input_width;

                  for(int kernel_row=0;kernel_row<kernel_size;kernel_row++) {
                      for(int kernel_col=0;kernel_col<kernel_size;kernel_col++) {

                          int new_row=row+kernel_row;
                          int new_col=col+kernel_col;

                          if(is_a_ge_zero_and_a_lt_b(new_row,input_height)&&is_a_ge_zero_and_a_lt_b(new_col,input_width)) {
                              *output_data++ = input_data[new_row*input_width+new_col];
                          } else {
                              *output_data++ = 0.0;
                          }            

                      }
                  }

              }
          }
      }    

  }


  template<typename T>
  void GManager<T>::LoadColDataIntoGpu(
    SharedArray<T> &_shared_array_buffer,
    T *input_data_buffer,
    int step_size,
    int group_size,
    int data_size)  {

    register T *input_data_buffer_ptr;
    register int pos = 0;
    int _size = 16 - data_size % 16;

    int n = data_size / 4;
    int _n = data_size % 4;

    for(int i=0;i<group_size;i++) {
      input_data_buffer_ptr = input_data_buffer + i;
      for(register int j=0;j<n;j+=4) {
        _shared_array_buffer[pos++] = *input_data_buffer_ptr;
        input_data_buffer_ptr = input_data_buffer_ptr + step_size;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr;
        input_data_buffer_ptr = input_data_buffer_ptr + step_size;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr;
        input_data_buffer_ptr = input_data_buffer_ptr + step_size;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr;
        input_data_buffer_ptr = input_data_buffer_ptr + step_size;
      }
      for(register int j=0;j<_n;j++) {
        _shared_array_buffer[pos++] = *input_data_buffer_ptr;
        input_data_buffer_ptr = input_data_buffer_ptr + step_size;
      }
      for(register int j=0;j<_size;j++) {
        _shared_array_buffer[pos++] = 0.0;
      }
    }
   
  }


  template<typename T>
  void GManager<T>::LoadDataIntoGpu(
    SharedArray<T> &_shared_array_buffer,
    T *input_data_buffer,
    int group_size,
    int data_size)  {

    register T *input_data_buffer_ptr = input_data_buffer;
    register int pos = 0;
    int _size = 16 - data_size % 16;

    int n = data_size / 4;
    int _n = data_size % 4;

    for(int i=0;i<group_size;i++) {
      for(register int j=0;j<n;j+=4) {
        _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
        _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
      }
      for(register int j=0;j<_n;j++) {
        _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
      }
      for(register int j=0;j<_size;j++) {
        _shared_array_buffer[pos++] = 0.0;
      }
    }
   
  }




  template<typename T>
  void GManager<T>::Init_Gpu_Memory() {
  	_gp_array[0].alloc(Max_GPU_Memory/3);
  	_gp_array[1].alloc(Max_GPU_Memory/3);
  	_gp_array[2].alloc(Max_GPU_Memory/3);
  }


  template<typename T>
  GManager<T>::GManager():Gpu_Memory_Basic_Block(350),Max_GPU_Memory(733409),GemmKernel(NULL) {
  	Init_Gpu_Memory();
  }

  extern GManager<float> global_gpu_manager;

  void gpu_gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k);

  GemmKernelType Get_GemmKernel();

  void Init_Kernel(GemmKernelType *Kernel);

};
#endif

#endif