cmake_minimum_required(VERSION 2.8.7)
project(PowerNet)
include_directories(include /usr/local/include)

if(GPU)
	set(QPU_DIR /home/pi/QPULib/QPULib/Lib)
	include_directories(${QPU_DIR}/Common/)
	include_directories(${QPU_DIR}/Source/)
	include_directories(${QPU_DIR}/VideoCore/)
	include_directories(${QPU_DIR}/Target/)
	include_directories(${QPU_DIR}/)
	file(GLOB_RECURSE QPU_H ${QPU_DIR}/Common/*.h ${QPU_DIR}/Source/*.h ${QPU_DIR}/Target/*.h ${QPU_DIR}/VideoCore/*.h ${QPU_DIR}/*.h)
	file(GLOB_RECURSE QPU_CPP ${QPU_DIR}/Common/*.cpp ${QPU_DIR}/Source/*.cpp ${QPU_DIR}/Target/*.cpp ${QPU_DIR}/VideoCore/*.cpp ${QPU_DIR}/*.cpp)
	ADD_LIBRARY(qpu-lib STATIC ${QPU_CPP} ${QPU_H})
endif(GPU)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)


# Add all headers and source files.
file(GLOB_RECURSE Power_CPP src/*.cpp third-party/*.cpp)
file(GLOB_RECURSE Power_H *.h)
list(REMOVE_ITEM Power_CPP main.cpp)


# include headers
include_directories(include/)
include_directories(third-party/)

# gen so
ADD_LIBRARY(Power SHARED ${Power_CPP} ${Power_H})

# gen static
ADD_LIBRARY(Power-static STATIC ${Power_CPP} ${Power_H})


ADD_EXECUTABLE(PowerTest test/main.cpp)
target_link_libraries(PowerTest Power-static)

if(GPU)
target_link_libraries(PowerTest qpu-lib)
endif(GPU)