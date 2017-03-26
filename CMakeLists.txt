cmake_minimum_required(VERSION 3.0)

project(simnets-tf)

set(SIMNETS_PYTHON_VERSION "3.5" CACHE STR "Python version to use")
option(SIMNETS_WITH_CUDA "Compile with CUDA enabled" 1)

find_package(PythonInterp ${SIMNETS_PYTHON_VERSION})

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++0x -fPIC  -Wall -D_GLIBCXX_USE_CXX11_ABI=0" ${CMAKE_CXX_FLAGS})
endif()

if (SIMNETS_WITH_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS -std=c++11 -DEIGEN_USE_GPU)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    RESULT_VARIABLE TF_RES
    OUTPUT_VARIABLE TF_INCLUDES)

if (NOT ${TF_RES} EQUAL 0)
    message(SEND_ERROR  "Can't import tensorflow")
endif()

include_directories(include
                    ${TF_INCLUDES})

add_compile_options(-DEIGEN_USE_THREADS)

if (SIMNETS_WITH_CUDA)
    cuda_add_library(simnet_ops SHARED src/similarity_op.cpp src/similarity_kernel_cpu.cpp
            include/ggemm_cpu.hpp
            include/similarity_kernel_common.hpp
            include/im2col.hpp
            src/im2col.cpp
            src/similarity_kernel_gpu.cu
            src/im2col.cu src/similarity_ref_op.cpp
            src/similarity_grad_kernel_cpu.cpp
            src/similarity_grad_kernel_gpu.cu
            src/mex_op.cpp
            src/mex_kernel_common.cpp
            src/mex_kernel_cpu.cpp)
else()
    add_library(simnet_ops SHARED src/similarity_op.cpp src/similarity_kernel_cpu.cpp
                                  include/ggemm_cpu.hpp
                                  include/similarity_kernel_common.hpp
                                  include/im2col.hpp
                                  src/im2col.cpp src/similarity_ref_op.cpp
                                  src/similarity_grad_kernel_cpu.cpp
                                  src/mex_op.cpp
                                  src/mex_kernel_common.cpp
                                  src/mex_kernel_cpu.cpp)
endif()

add_custom_target(test_simnet_ops
        COMMAND cd $<TARGET_FILE_DIR:simnet_ops>
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/test/test.py)

set_target_properties(simnet_ops PROPERTIES PREFIX "")
target_include_directories(simnet_ops PUBLIC ${TF_INCLUDES})
