cmake_minimum_required(VERSION 3.29)
project(CudaCode CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(CudaCode main.cu)

set_target_properties(CudaCode PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
