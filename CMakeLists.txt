cmake_minimum_required(VERSION 3.0)

project(simnets-tf)

set(SIMNETS_PYTHON_VERSION "3.5" CACHE STR "Python version to use")
option(SIMNETS_WITH_CUDA "Compile with CUDA enabled" 1)
option(SIMNETS_OLD_CXX_ABI "Compile with old ABI change this if you get unresolved symbols" 1)

find_package(PythonInterp ${SIMNETS_PYTHON_VERSION})

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++0x -fPIC  -Wall" ${CMAKE_CXX_FLAGS})
    if(SIMNETS_OLD_CXX_ABI)
        set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
    endif()
endif()

if (SIMNETS_WITH_CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS -std=c++11 -DSIMNETS_WITH_CUDA=1 -DEIGEN_USE_GPU --expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=\"sm_30,compute_30\";-gencode arch=compute_35,code=\"sm_35,compute_35\";-gencode arch=compute_52,code=\"sm_52,compute_52\")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_include())"
    RESULT_VARIABLE TF_RES
    OUTPUT_VARIABLE TF_INCLUDES)

if (NOT ${TF_RES} EQUAL 0)
    message(SEND_ERROR  "Can't import tensorflow")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib())"
RESULT_VARIABLE TF_RES
OUTPUT_VARIABLE TF_LIBS_DIR)

include_directories(include
                    third_party/maps/include
                    ${TF_INCLUDES}
                    "${TF_INCLUDES}/external/nsync/public")

link_directories(${TF_LIBS_DIR})

add_compile_options(-DEIGEN_USE_THREADS)

if (SIMNETS_WITH_CUDA)
    add_compile_options(-DSIMNETS_WITH_CUDA=1)
endif()

add_library(mex_dims_helper SHARED src/kernels/mex_dims_helper.cpp
        src/kernels/mex_kernel_common.cpp)

if (SIMNETS_WITH_CUDA)
    cuda_add_library(simnet_ops SHARED src/ops/similarity_op.cpp src/kernels/similarity_kernel_cpu.cpp
            include/utils/ggemm_cpu.hpp
            include/kernels/similarity_kernel_common.hpp
            include/utils/im2col.hpp
            src/utils/im2col.cpp
            src/kernels/similarity_kernel_gpu.cu
            src/utils/im2col.cu src/kernels/similarity_ref_kernel_cpu.cpp
            src/kernels/similarity_grad_kernel_cpu.cpp
            src/kernels/similarity_grad_kernel_gpu.cu
            src/ops/mex_op.cpp
            src/kernels/mex_kernel_common.cpp
            src/kernels/mex_kernel_cpu.cpp
            src/kernels/mex_kernel_gpu.cu
            src/kernels/mex_grad_kernel_cpu.cpp
            src/kernels/mex_grad_kernel_gpu.cu)
else()
    add_library(simnet_ops SHARED src/ops/similarity_op.cpp src/kernels/similarity_kernel_cpu.cpp
            include/utils/ggemm_cpu.hpp
            include/kernels/similarity_kernel_common.hpp
            include/utils/im2col.hpp
            src/utils/im2col.cpp src/kernels/similarity_ref_kernel_cpu.cpp
            src/kernels/similarity_grad_kernel_cpu.cpp
            src/ops/mex_op.cpp
            src/kernels/mex_kernel_common.cpp
            src/kernels/mex_kernel_cpu.cpp
            src/kernels/mex_grad_kernel_cpu.cpp)
endif()

target_link_libraries(mex_dims_helper tensorflow_framework)
target_link_libraries(simnet_ops tensorflow_framework)

add_custom_target(test_simnet_ops
        ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_INSTALL_PREFIX} TF_CPP_MIN_LOG_LEVEL=3 ${PYTHON_EXECUTABLE} -m unittest simnets.test
        DEPENDS clion_install)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.util import get_platform; import sys; sys.stdout.write(get_platform())"
        OUTPUT_VARIABLE PY_PLATFORM)
        
string(REPLACE "." "" PY_VER_NO_DOT ${SIMNETS_PYTHON_VERSION})

set(BDIST_WHEEL_PARAMS bdist_wheel --plat-name=${PY_PLATFORM} --python-tag="cp${PY_VER_NO_DOT}")

add_custom_target(create_wheel
        ${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/setup.py ${BDIST_WHEEL_PARAMS}
        WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}
        DEPENDS clion_install)

install(DIRECTORY python/simnets DESTINATION ".")
install(FILES python/setup.py README.md
        DESTINATION ".")
install(FILES $<TARGET_FILE:simnet_ops> $<TARGET_FILE:mex_dims_helper>
        DESTINATION "simnets/ops")

if(SIMNETS_WITH_CUDA)
    install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/.compile_mode \"gpu\")")
else()
    install(CODE "file(WRITE ${CMAKE_INSTALL_PREFIX}/.compile_mode \"cpu\")")
endif()

set_target_properties(simnet_ops PROPERTIES PREFIX "")
target_include_directories(simnet_ops PUBLIC ${TF_INCLUDES})

add_custom_target(clion_install
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install)
