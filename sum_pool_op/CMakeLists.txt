cmake_minimum_required(VERSION 3.0)

project(sum_pool_op)

set(SUMPOOL_PYTHON_VERSION "3.5" CACHE STR "Python version to use")

find_package(PythonInterp ${SUMPOOL_PYTHON_VERSION})

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++0x -fPIC  -Wall -D_GLIBCXX_USE_CXX11_ABI=0" ${CMAKE_CXX_FLAGS})
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    RESULT_VARIABLE TF_RES
    OUTPUT_VARIABLE TF_INCLUDES)

if (NOT ${TF_RES} EQUAL 0)
    message(SEND_ERROR  "Can't import tensorflow")
endif()

add_library(sum_pool_op SHARED src/sum_pool_op.cpp src/sum_pool_kernel.cpp)
add_custom_target(sum_pool_test
        COMMAND ${PYTHON_EXECUTABLE} sum_pool_test.py $<TARGET_FILE:sum_pool_op>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

set_target_properties(sum_pool_op PROPERTIES PREFIX "")
target_include_directories(sum_pool_op PUBLIC ${TF_INCLUDES})
