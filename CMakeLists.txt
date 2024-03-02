cmake_minimum_required(VERSION 3.18)
project(myapp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /openmp /arch:AVX2")
else()
    set(CUDAToolkit_ROOT "/usr/local/cuda-11.8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2")
    //set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp gpu_mul.cu)
target_link_libraries(${PROJECT_NAME} CUDA::cudart OpenMP::OpenMP_CXX)
