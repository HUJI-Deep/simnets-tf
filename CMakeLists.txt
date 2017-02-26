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
    set(CUDA_NVCC_FLAGS -std=c++11)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    RESULT_VARIABLE TF_RES
    OUTPUT_VARIABLE TF_INCLUDES)

if (NOT ${TF_RES} EQUAL 0)
    message(SEND_ERROR  "Can't import tensorflow")
endif()

include_directories(${TF_INCLUDES})

if (SIMNETS_WITH_CUDA)
    cuda_add_library(simnet_ops SHARED src/similarity_op.cpp src/similarity_kernel_cpu.cpp src/similarity_kernel_gpu.cu)
else()
    add_library(simnet_ops SHARED src/similarity_op.cpp src/similarity_kernel_cpu.cpp)
endif()

#add_custom_target(sum_pool_test
#        COMMAND ${PYTHON_EXECUTABLE} sum_pool_test.py $<TARGET_FILE:sum_pool_op>
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

set_target_properties(simnet_ops PROPERTIES PREFIX "")
target_include_directories(simnet_ops PUBLIC ${TF_INCLUDES})

