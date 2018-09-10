
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <alloca.h>
#include <string.h>
#include <stdlib.h>

class GPGemm {
public:
    GPGemm();
    ~GPGemm();

    void displayInfo();

    void createPlatforms();
    void createDevices();
    void createContext();
    void createBuildProgram(std::string source);
    void createKernel(std::string kernel_func);
    cl_mem createBuffer(cl_mem_flags flags,size_t size,void *host_ptr);
    void createCommandQueue(int deviceIndex=0);

    void setKernelArg(cl_uint index,size_t arg_size,const void *arg_value);

    void setMatrixFloat(int M,int N,int K);
    void setMatrixBit(int M,int N,int K);

    void EnqueueNDRangeKernel(cl_uint dim);
    void EnqueueWriteBuffer(cl_mem buffer,size_t cb,const void *ptr);
    void EnqueueReadBuffer(cl_mem buffer,size_t cb,void *ptr);

    void preRuntimeInit(std::string source,std::string func);
    void preComputeFloat(int M,int N,int K);
    void gpu_gemm_float(float *A,float *B,float *C,size_t *g_size,size_t *l_size);
    void preComputeBit(int M,int N,int K);
    void gpu_gemm_bit(char *A,char *B,char *C,size_t *g_size,size_t *l_size);

    void preLoad();
    void clean();

    void setGlobalWorkSize(size_t *ptr) {
        globalWorkSize=ptr;
    }

    void setLocalWorkSize(size_t *ptr) {
        localWorkSize=ptr;
    }

private:
    static int cur_gemms_count;
    static bool pre_load_flag;

    static cl_int curPlatformIndex;
    static cl_uint numPlatforms;
    static cl_platform_id *platformIds;
    static cl_uint numDevices;
    static cl_device_id *deviceIds;
    static cl_context context;

    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;

    size_t *globalWorkSize;
    size_t *localWorkSize;

    cl_mem MatA;
    cl_mem MatB;
    cl_mem MatC;

    int Mdim;
    int Ndim;
    int Kdim;

};




    int GPGemm::cur_gemms_count=0;
    bool GPGemm::pre_load_flag=false;
    cl_int GPGemm::curPlatformIndex=0;
    cl_uint GPGemm::numPlatforms=0;
    cl_platform_id* GPGemm::platformIds=NULL;
    cl_uint GPGemm::numDevices=0;
    cl_device_id* GPGemm::deviceIds=NULL;
    cl_context GPGemm::context=NULL;

    GPGemm::GPGemm() {
        cur_gemms_count++;

        if(!pre_load_flag) {
            preLoad();
            pre_load_flag=true;
        }

    }

    GPGemm::~GPGemm() {
        cur_gemms_count--;
        clean();
    }


    void GPGemm::clean() {
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseMemObject(MatA);
        clReleaseMemObject(MatB);
        clReleaseMemObject(MatC);
        clReleaseCommandQueue(queue);

        if(cur_gemms_count==0) {
            clReleaseContext(context);
            if(platformIds!=NULL) {
                free(platformIds);
            }
            if(deviceIds!=NULL) {
                free(deviceIds);
            }
        }
    }

    void GPGemm::preLoad() {
        createPlatforms();
        createDevices();
        createContext();
    }

    void GPGemm::setMatrixFloat(int M,int N,int K) {
        int szA=M*K;
        int szB=K*N;
        int szC=M*N;
        Mdim=M; Ndim=N; Kdim=K;
        MatA=createBuffer(CL_MEM_READ_ONLY,sizeof(float)*szA,NULL);
        MatB=createBuffer(CL_MEM_READ_ONLY,sizeof(float)*szB,NULL);
        MatC=createBuffer(CL_MEM_WRITE_ONLY,sizeof(float)*szC,NULL);
    }


    void GPGemm::setMatrixBit(int M,int N,int K) {
        int szA=M*K;
        int szB=K*N;
        int szC=M*N;
        Mdim=M; Ndim=N; Kdim=K;
        MatA=createBuffer(CL_MEM_READ_ONLY,sizeof(char)*szA,NULL);
        MatB=createBuffer(CL_MEM_READ_ONLY,sizeof(char)*szB,NULL);
        MatC=createBuffer(CL_MEM_WRITE_ONLY,sizeof(char)*szC,NULL);
    }


    void GPGemm::preComputeFloat(int M,int N,int K) {
        setMatrixFloat(M,N,K);
        setKernelArg(0,sizeof(int),&Mdim);
        setKernelArg(1,sizeof(int),&Ndim);
        setKernelArg(2,sizeof(int),&Kdim);
        setKernelArg(3,sizeof(cl_mem),&MatA);
        setKernelArg(4,sizeof(cl_mem),&MatB);
        setKernelArg(5,sizeof(cl_mem),&MatC);
    }

    void GPGemm::gpu_gemm_float(float *A,float *B,float *C,size_t *g_size,size_t *l_size) {
        int szA=Mdim*Kdim;
        int szB=Kdim*Ndim;
        int szC=Mdim*Ndim;
        EnqueueWriteBuffer(MatA,sizeof(float)*szA,A);
        EnqueueWriteBuffer(MatB,sizeof(float)*szB,B);
        setGlobalWorkSize(g_size);
        setLocalWorkSize(l_size);
        EnqueueNDRangeKernel(1);
        clFinish(queue);
        EnqueueReadBuffer(MatC,sizeof(float)*szC,C);
    }

    void GPGemm::preComputeBit(int M,int N,int K) {
        setMatrixBit(M,N,K);
        setKernelArg(0,sizeof(int),&Mdim);
        setKernelArg(1,sizeof(int),&Ndim);
        setKernelArg(2,sizeof(int),&Kdim);
        setKernelArg(3,sizeof(cl_mem),&MatA);
        setKernelArg(4,sizeof(cl_mem),&MatB);
        setKernelArg(5,sizeof(cl_mem),&MatC);
    }


    void GPGemm::gpu_gemm_bit(char *A,char *B,char *C,size_t *g_size,size_t *l_size) {
        int szA=Mdim*Kdim;
        int szB=Kdim*Ndim;
        int szC=Mdim*Ndim;
        EnqueueWriteBuffer(MatA,sizeof(char)*szA,A);
        EnqueueWriteBuffer(MatB,sizeof(char)*szB,B);
        setGlobalWorkSize(g_size);
        setLocalWorkSize(l_size);
        EnqueueNDRangeKernel(2);
        clFinish(queue);
        EnqueueReadBuffer(MatC,sizeof(char)*szC,C);
    }


    void GPGemm::preRuntimeInit(std::string source,std::string func) {
        createBuildProgram(source);
        createKernel(func);
        createCommandQueue();
    }


    void GPGemm::EnqueueReadBuffer(cl_mem buffer,size_t cb,void *ptr) {
        cl_int status;

        status=clEnqueueReadBuffer(queue,buffer,CL_TRUE,0,cb,ptr,0,NULL,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to Enqueue Read Buffer."<<std::endl;
            exit(1);
        }
    }

    void GPGemm::EnqueueWriteBuffer(cl_mem buffer,size_t cb,const void *ptr) {
        cl_int status;

        status=clEnqueueWriteBuffer(queue,buffer,CL_TRUE,0,cb,ptr,0,NULL,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to Enqueue Write Buffer."<<std::endl;
            exit(1);
        }
    }



    void GPGemm::createPlatforms() {
        cl_int status;

        status=clGetPlatformIDs(0,NULL,&numPlatforms);
        if(status!=CL_SUCCESS||numPlatforms<0) {
            std::cerr<<"Failed to find any OpenCL platforms."<<std::endl;
            exit(1);
        }

        platformIds=(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
        if(platformIds==NULL) {
            std::cerr<<"Failed allocate memory for platformIds."<<std::endl;
            exit(1);
        }

        status=clGetPlatformIDs(numPlatforms,platformIds,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to create OpenCL platforms."<<std::endl;
            exit(1);
        }
    }

    void GPGemm::createDevices() {
        cl_int status;

        cl_int cnt;
        for(cnt=0; cnt<numPlatforms; cnt++) {
            status=clGetDeviceIDs(platformIds[cnt],CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);
            if(numDevices<1) {
                std::cerr<<"No GPU Device Found."<<std::endl;
                exit(1);
            }
            deviceIds=(cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
            if(deviceIds==NULL) {
                std::cerr<<"Failed allocate memory for deviceIds."<<std::endl;
                exit(1);
            }
            status=clGetDeviceIDs(platformIds[cnt],CL_DEVICE_TYPE_GPU,1,deviceIds,NULL);
            if(status!=CL_SUCCESS) {
                std::cerr<<"clGetDeviceIDs Error."<<std::endl;
                exit(1);
            }
            break;
        }
        curPlatformIndex=cnt;
    }

    void GPGemm::createContext() {
        cl_int status;

        cl_context_properties prop[]={
            CL_CONTEXT_PLATFORM,(cl_context_properties)platformIds[curPlatformIndex],0};

        context=clCreateContext(
            prop,numDevices,deviceIds,NULL,NULL,&status);
        if(status!=CL_SUCCESS) {
            std::cerr<<"clCreateContext Error."<<std::endl;
            exit(1);
        }
    }

    void GPGemm::createBuildProgram(std::string source) {
        cl_int status;

        std::ifstream srcFile(source.c_str());
        if(!srcFile.is_open()) {
            std::cerr<<"Open Source File ERROR"<<std::endl;
            exit(1);
        }

        std::string srcProg(std::istreambuf_iterator<char>(srcFile),(std::istreambuf_iterator<char>()));

        const char *src=srcProg.c_str();
        size_t length=srcProg.length();

        program=clCreateProgramWithSource(context,1,&src,&length,&status);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Create Program ERROR"<<std::endl;
            exit(1);
        }

        status=clBuildProgram(program,numDevices,deviceIds,NULL,NULL,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Build Program ERROR"<<std::endl;
            if(status==CL_BUILD_PROGRAM_FAILURE) {
                char buildLog[1000];
                clGetProgramBuildInfo(program,deviceIds[0],CL_PROGRAM_BUILD_LOG,sizeof(buildLog),buildLog,NULL);
                std::cerr<<"Error in kernel: "<<std::endl;
                std::cerr<<buildLog;
            }
            exit(1);
        }
    }

    void GPGemm::createKernel(std::string kernel_func) {
        cl_int status;

        kernel=clCreateKernel(program,kernel_func.c_str(),&status);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Create Kernel ERROR"<<std::endl;
            exit(1);
        }
    }

    cl_mem GPGemm::createBuffer(cl_mem_flags flags,size_t size,void *host_ptr) {
        cl_int status;
        cl_mem buffer=clCreateBuffer(context,flags,size,host_ptr,&status);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Create Buffer ERROR"<<std::endl;
            exit(1);
        }
        return buffer;
    }

    void GPGemm::createCommandQueue(int deviceIndex) {
        cl_int status;

        if(deviceIndex>=numDevices) {
            std::cerr<<"DeviceIndex Out of Range."<<std::endl;
            exit(1);
        }
        queue=clCreateCommandQueue(context,deviceIds[deviceIndex],0,&status);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Create Command Queue ERROR."<<std::endl;
            exit(1);
        }
    }


    void GPGemm::setKernelArg(cl_uint index,size_t arg_size,const void *arg_value) {
        cl_int status;

        status=clSetKernelArg(kernel,index,arg_size,arg_value);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to Set Kernel Args."<<std::endl;
            exit(1);
        }
    }

    void GPGemm::EnqueueNDRangeKernel(cl_uint dim) {
        cl_int status;
        cl_uint ndim=dim;

        status=clEnqueueNDRangeKernel(queue,kernel,ndim,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to Enqueue NDRange Kernel."<<status<<std::endl;
            exit(1);
        }
    }


    void GPGemm::displayInfo() {}

    void init_mat(float *A,float *B,float *C,int M,int N,int K) {
        int szA=M*K;
        int szB=K*N;
        int szC=M*N;

        for(int i=0;i<szA;i++) {
            A[i]=rand()%100;
        }

        for(int i=0;i<szB;i++) {
            B[i]=rand()%100;
        }

        for(int i=0;i<szC;i++) {
            C[i]=0;
        }
    }

    void gemm_vaild(float *A,float *B,float *C,int M,int N,int K) {
        for(int i=0;i<M;i++) {
            for(int j=0;j<N;j++) {
                float sum=0;
                for(int k=0;k<K;k++) {
                    sum+=A[i*K+k]*B[k*N+j];
                }
                C[i*N+j]=sum;
            }
        }
    }

    bool result_compare(float *C,float *E,int size) {
        for(int i=0;i<size;i++) {
            if(int(C[i])!=int(E[i])) {
                return false;
            }
        }
        return true;
    }

    void print_matrix(float *A,int M,int N) {
        int szA=M*N;
        for(int i=0;i<szA;i++) {
            std::cout<<A[i];
            if((i+1)%N==0) {
                std::cout<<std::endl;
            } else {
                std::cout<<" ";
            }
        }
    }

    void print_debug(float *A,float *B,float *C,float *E,int M,int N,int K) {
        std::cout<<"Matrix A:\n";
        print_matrix(A,M,K);
        std::cout<<"Matrix B:\n";
        print_matrix(B,K,N);
        std::cout<<"Matrix C:\n";
        print_matrix(C,M,N);
        std::cout<<"Matrix E:\n";
        print_matrix(E,M,N);
    }

    int main() {
        int dim=16;
        int M,N,K;
        M=N=K=dim;
        float A[M*K];
        float B[K*N];
        float C[M*N];
        float E[M*N];

        size_t global[]={M};
        size_t local[]={2};
        init_mat(A,B,C,M,N,K);
        memcpy(E,C,sizeof(float)*M*N);

        GPGemm gp;
        gp.preRuntimeInit("/home/pi/gemm_float.cl","gemm_float");
        gp.preComputeFloat(M,N,K);
        gp.gpu_gemm_float(A,B,C,global,local);

        gemm_vaild(A,B,E,M,N,K);
        if(result_compare(C,E,M*N)) {
            std::cout<<"Result Correct."<<std::endl;
        } else {
            std::cout<<"Result Error."<<std::endl;
            print_debug(A,B,C,E,M,N,K);
        }
    }
