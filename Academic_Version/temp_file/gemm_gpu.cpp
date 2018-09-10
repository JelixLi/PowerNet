#include "math/gpu_gemm.h"
#include <iostream>

namespace mdl {

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
        EnqueueNDRangeKernel(2);
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

    void GPGemm::EnqueueNDRangeKernel(int dim) {
        cl_int status;

        status=clEnqueueNDRangeKernel(queue,kernel,dim,NULL,globalWorkSize,localWorkSize,0,NULL,NULL);
        if(status!=CL_SUCCESS) {
            std::cerr<<"Failed to Enqueue NDRange Kernel."<<std::endl;
            exit(1);
        }
    }


    void GPGemm::displayInfo()
    {
        cl_int errNum;
        cl_uint numPlatforms;
        cl_platform_id * platformIds;
        cl_context context = NULL;

        // First, query the total number of platforms
        errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
        if (errNum != CL_SUCCESS || numPlatforms <= 0)
        {
            std::cerr << "Failed to find any OpenCL platform." << std::endl;
            return;
        }

        // Next, allocate memory for the installed plaforms, and qeury
        // to get the list.
        platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
        // First, query the total number of platforms
        errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to find any OpenCL platforms." << std::endl;
            return;
        }

        std::cout << "Number of platforms: \t" << numPlatforms << std::endl;
        // Iterate through the list of platforms displaying associated information
        for (cl_uint i = 0; i < numPlatforms; i++) {
            // First we display information associated with the platform
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_PROFILE,
                "CL_PLATFORM_PROFILE");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_VERSION,
                "CL_PLATFORM_VERSION");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_VENDOR,
                "CL_PLATFORM_VENDOR");
            DisplayPlatformInfo(
                platformIds[i],
                CL_PLATFORM_EXTENSIONS,
                "CL_PLATFORM_EXTENSIONS");

            // Now query the set of devices associated with the platform
            cl_uint numDevices;
            errNum = clGetDeviceIDs(
                platformIds[i],
                CL_DEVICE_TYPE_ALL,
                0,
                NULL,
                &numDevices);
            if (errNum != CL_SUCCESS)
            {
                std::cerr << "Failed to find OpenCL devices." << std::endl;
                return;
            }

            cl_device_id * devices = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
            errNum = clGetDeviceIDs(
                platformIds[i],
                CL_DEVICE_TYPE_ALL,
                numDevices,
                devices,
                NULL);
            if (errNum != CL_SUCCESS)
            {
                std::cerr << "Failed to find OpenCL devices." << std::endl;
                return;
            }

            std::cout << "\tNumber of devices: \t" << numDevices << std::endl;
            // Iterate through each device, displaying associated information
            for (cl_uint j = 0; j < numDevices; j++)
            {
                InfoDevice<cl_device_type>::display(
                    devices[j],
                    CL_DEVICE_TYPE,
                    "CL_DEVICE_TYPE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_VENDOR_ID,
                    "CL_DEVICE_VENDOR_ID");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_COMPUTE_UNITS,
                    "CL_DEVICE_MAX_COMPUTE_UNITS");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");

                InfoDevice<ArrayType<size_t> >::display(
                    devices[j],
                    CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    "CL_DEVICE_MAX_WORK_ITEM_SIZES");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    "CL_DEVICE_MAX_WORK_GROUP_SIZE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

    #ifdef CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                    "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                    "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF");
    #endif

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    "CL_DEVICE_MAX_CLOCK_FREQUENCY");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_ADDRESS_BITS,
                    "CL_DEVICE_ADDRESS_BITS");

                InfoDevice<cl_ulong>::display(
                    devices[j],
                    CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    "CL_DEVICE_MAX_MEM_ALLOC_SIZE");

                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_IMAGE_SUPPORT,
                    "CL_DEVICE_IMAGE_SUPPORT");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_READ_IMAGE_ARGS,
                    "CL_DEVICE_MAX_READ_IMAGE_ARGS");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                    "CL_DEVICE_MAX_WRITE_IMAGE_ARGS");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE2D_MAX_WIDTH,
                    "CL_DEVICE_IMAGE2D_MAX_WIDTH");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE2D_MAX_WIDTH,
                    "CL_DEVICE_IMAGE2D_MAX_WIDTH");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                    "CL_DEVICE_IMAGE2D_MAX_HEIGHT");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE3D_MAX_WIDTH,
                    "CL_DEVICE_IMAGE3D_MAX_WIDTH");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                    "CL_DEVICE_IMAGE3D_MAX_HEIGHT");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_IMAGE3D_MAX_DEPTH,
                    "CL_DEVICE_IMAGE3D_MAX_DEPTH");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_SAMPLERS,
                    "CL_DEVICE_MAX_SAMPLERS");

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_MAX_PARAMETER_SIZE,
                    "CL_DEVICE_MAX_PARAMETER_SIZE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                    "CL_DEVICE_MEM_BASE_ADDR_ALIGN");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                    "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");

                InfoDevice<cl_device_fp_config>::display(
                    devices[j],
                    CL_DEVICE_SINGLE_FP_CONFIG,
                    "CL_DEVICE_SINGLE_FP_CONFIG");

                InfoDevice<cl_device_mem_cache_type>::display(
                    devices[j],
                    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                    "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                    "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");

                InfoDevice<cl_ulong>::display(
                    devices[j],
                    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                    "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");

                InfoDevice<cl_ulong>::display(
                    devices[j],
                    CL_DEVICE_GLOBAL_MEM_SIZE,
                    "CL_DEVICE_GLOBAL_MEM_SIZE");

                InfoDevice<cl_ulong>::display(
                    devices[j],
                    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                    "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");

                InfoDevice<cl_uint>::display(
                    devices[j],
                    CL_DEVICE_MAX_CONSTANT_ARGS,
                    "CL_DEVICE_MAX_CONSTANT_ARGS");

                InfoDevice<cl_device_local_mem_type>::display(
                    devices[j],
                    CL_DEVICE_LOCAL_MEM_TYPE,
                    "CL_DEVICE_LOCAL_MEM_TYPE");

                InfoDevice<cl_ulong>::display(
                    devices[j],
                    CL_DEVICE_LOCAL_MEM_SIZE,
                    "CL_DEVICE_LOCAL_MEM_SIZE");

                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                    "CL_DEVICE_ERROR_CORRECTION_SUPPORT");

    #ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    "CL_DEVICE_HOST_UNIFIED_MEMORY");
    #endif

                InfoDevice<std::size_t>::display(
                    devices[j],
                    CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                    "CL_DEVICE_PROFILING_TIMER_RESOLUTION");

                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_ENDIAN_LITTLE,
                    "CL_DEVICE_ENDIAN_LITTLE");

                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_AVAILABLE,
                    "CL_DEVICE_AVAILABLE");

                InfoDevice<cl_bool>::display(
                    devices[j],
                    CL_DEVICE_COMPILER_AVAILABLE,
                    "CL_DEVICE_COMPILER_AVAILABLE");

                InfoDevice<cl_device_exec_capabilities>::display(
                    devices[j],
                    CL_DEVICE_EXECUTION_CAPABILITIES,
                    "CL_DEVICE_EXECUTION_CAPABILITIES");

                InfoDevice<cl_command_queue_properties>::display(
                    devices[j],
                    CL_DEVICE_QUEUE_PROPERTIES,
                    "CL_DEVICE_QUEUE_PROPERTIES");

                InfoDevice<cl_platform_id>::display(
                    devices[j],
                    CL_DEVICE_PLATFORM,
                    "CL_DEVICE_PLATFORM");

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_NAME,
                    "CL_DEVICE_NAME");

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_VENDOR,
                    "CL_DEVICE_VENDOR");

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DRIVER_VERSION,
                    "CL_DRIVER_VERSION");

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_PROFILE,
                    "CL_DEVICE_PROFILE");

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_VERSION,
                    "CL_DEVICE_VERSION");

    #ifdef CL_DEVICE_OPENCL_C_VERSION
                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_OPENCL_C_VERSION,
                    "CL_DEVICE_OPENCL_C_VERSION");
    #endif

                InfoDevice<ArrayType<char> >::display(
                    devices[j],
                    CL_DEVICE_EXTENSIONS,
                    "CL_DEVICE_EXTENSIONS");


                std::cout << std::endl << std::endl;
            }
        }
    }
}
