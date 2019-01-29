#!/usr/bin/env sh

build_linux_fn() {
    if [ ! `which protoc` ]; then
        echo "please install the latest protobuf using homebrew"
        return
        # echo "installing protobuf."
        # brew install protobuf
        # if [ ! $? ]; then
        #     echo "protobuf install failed."
        #     return
        # fi
    fi
    if [ ! `which cmake` ]; then
        echo "installing cmake."
        brew install cmake
        if [ ! $? ]; then
            echo "cmake install failed."
            return
        fi
    fi
    PLATFORM="x86"
    MODE="Release"
    CXX_FLAGS="-g -std=c++11"
    LD_FLAGS="-pthread "
    BUILD_DIR=build/release/"${PLATFORM}"
    mkdir -p ${BUILD_DIR}/build
    cp -r test/model ${BUILD_DIR}/build
    cd "${BUILD_DIR}"
    CMAKE="cmake"
    "${CMAKE}" ../../.. \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LD_FLAGS}" 

    make -j 8
}


build_gpu_fn() {
    if [ ! `which protoc` ]; then
        echo "please install the latest protobuf using homebrew"
        return
        # echo "installing protobuf."
        # brew install protobuf
        # if [ ! $? ]; then
        #     echo "protobuf install failed."
        #     return
        # fi
    fi
    if [ ! `which cmake` ]; then
        echo "installing cmake."
        brew install cmake
        if [ ! $? ]; then
            echo "cmake install failed."
            return
        fi
    fi
    PLATFORM="x86"
    MODE="Release"
    CXX_FLAGS="-g -std=c++14 -DGPU -DQPU_MODE"
    LD_FLAGS="-pthread "
    BUILD_DIR=build/release/"${PLATFORM}"
    mkdir -p ${BUILD_DIR}/build
    cp -r test/model ${BUILD_DIR}/build
    cd "${BUILD_DIR}"
    CMAKE="cmake"
    "${CMAKE}" ../../.. \
        -DCMAKE_BUILD_TYPE="${MODE}" \
        -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LD_FLAGS}" \
        -DGPU=true

    make -j 8
}


error_fn () {
    echo "unknown argument"
}

if [ $# = 0 ]; then
    echo "error: target missing!"
    echo "available targets: mac|linux|ios|android"
    echo "sample usage: ./build.sh mac"
else
    if [ $1 = "linux" ]; then
        build_linux_fn
    elif [ $1 = "gpu" ]; then
        build_gpu_fn
    else
        error_fn
    fi
fi

