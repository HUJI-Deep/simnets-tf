/**
 *  Generalization of Matrix Multiplication (C = A * B).
 *  The original operation is defined by $ C_{ij} = \sum_{k = 1}^K A_{ik} * B_{kj} $.
 *  The generalized operation supports swapping the multiplication and addition opertaions above
 *  with a generation combine_function(a, b) and an accumilation_function(acc_c, c). For example,
 *  one can define the operation by $ C_{ij} = \max_{k = 1}^K (A_{ik} +  B_{kj}) $.
 *
 *  A farther generalization is allowing the combine_function to access the matching element from C,
 *  i.e., $ C_{ij} = \textrm{accumlate}_{k = 1}^K \textrm{combine}(A_{ik}, B_{kj}, C_{ij})$. This
 *  allows one to implement parametrized softmax (i.e., log(sum(exp(a_i + b_i)))), or the backward
 *  steps of many convolutional-like network layers.
 */

// #define QUICK_COMPILE
#ifndef _GGEMM_H_
#define _GGEMM_H_
#include "ggemm_cpu.hpp"
#include "maps/maps.cuh"
#define DEFAULT_BLOCK_SIZE (16)
#define DEFAULT_VECTOR_SIZE (8)
#if __CUDA_ARCH__ >= 200
const int GGEMM_CUDA_NUM_THREADS = 1024;
#else
const int GGEMM_CUDA_NUM_THREADS = 512;
#endif

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition;\
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
// CUDA: number of blocks for threads.
inline int GGEMM_GET_BLOCKS(const int N) {
    return (N + GGEMM_CUDA_NUM_THREADS - 1) / GGEMM_CUDA_NUM_THREADS;
}

template<typename Dtype, typename Ptype, Dtype (*APPLY_F)(Dtype, Ptype)>
__global__ void array_func_kernel(const int n, const Ptype param, const Dtype* x, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = APPLY_F(x[index], param);
    }
}
template<typename Dtype, typename Ptype, Dtype (*APPLY_F)(Dtype, Dtype, Ptype)>
__global__ void array_func_kernel(const int n, const Ptype param, const Dtype* x1, const Dtype* x2, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = APPLY_F(x1[index], x2[index], param);
    }
}

template<typename Dtype, typename Ptype, Dtype (*APPLY_F)(Dtype, Ptype)>
void caffe_gpu_array_func(const int n, const Ptype param, const Dtype* x, Dtype* y) {
    CHECK_GT(n, 0);
    array_func_kernel<Dtype, Ptype, APPLY_F><<<GGEMM_GET_BLOCKS(n), GGEMM_CUDA_NUM_THREADS>>>(
            n, param, x, y);
}
template<typename Dtype, typename Ptype, Dtype (*APPLY_F)(Dtype, Dtype, Ptype)>
void caffe_gpu_array_func(const int n, const Ptype param, const Dtype* x1, const Dtype* x2, Dtype* y) {
    CHECK_GT(n, 0);
    array_func_kernel<Dtype, Ptype, APPLY_F><<<GGEMM_GET_BLOCKS(n), GGEMM_CUDA_NUM_THREADS>>>(
            n, param, x1, x2, y);
}

template <typename Dtype> __global__
void interlace_kernel(const int N, const Dtype* a, const Dtype* b, typename vec<Dtype>::vec2* o) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        o[i] = make_vec2<Dtype>(a[i], b[i]);
    }
}
template <typename Dtype> __global__
void interlace_kernel(const int N,
                      const Dtype* a, const Dtype* b, const Dtype* c, const Dtype* d, typename vec<Dtype>::vec4* o) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        o[i] = make_vec4<Dtype>(a[i], b[i], c[i], d[i]);
    }
}
template <typename Dtype>
void interlace_gpu(const int N, const Dtype* a, const Dtype* b, typename vec<Dtype>::vec2* o,
                   const cudaStream_t stream = 0) {
    interlace_kernel<Dtype><<<GGEMM_GET_BLOCKS(N), GGEMM_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, o);
}
template <typename Dtype>
void interlace_gpu(const int N, const Dtype* a, const Dtype* b, const Dtype* c, const Dtype* d, typename vec<Dtype>::vec4* o,
                   const cudaStream_t stream = 0) {
    interlace_kernel<Dtype><<<GGEMM_GET_BLOCKS(N), GGEMM_CUDA_NUM_THREADS, 0, stream>>>(N, a, b, c, d, o);
}

template <typename Dtype>
__global__ void deinterlace_vec2_kernel(const int N,
                                        const typename vec<Dtype>::vec2* in, Dtype* a, Dtype* b) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        const typename vec<Dtype>::vec2 data = in[i];
        a[i] = data.x;
        b[i] = data.y;
    }
}
template <typename Dtype> void deinterlace_gpu(const int N,
                                               const typename vec<Dtype>::vec2* in, Dtype* a, Dtype* b) {
    deinterlace_vec2_kernel<Dtype><<<GGEMM_GET_BLOCKS(N), GGEMM_CUDA_NUM_THREADS>>>(N, in, a, b);
}

template <int BLOCK_WIDTH, int BLOCK_HEIGHT, int ILP_X, int ILP_Y,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
__global__ void ggemm_kernel(
        maps::BlockSingleGPU<Atype, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1, maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> A,
        maps::BlockSingleGPU<Btype, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1, maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> B,
        maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries, ILP_X, ILP_Y, 1, maps::DirectIO> Cin,
        maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C,
        const Ctype Cinit, const Ptype extra_params, const int batch_size = 1,
        const int A_batch_stride = -1, const int B_batch_stride = -1, const int C_batch_stride = -1) {

    __shared__ typename decltype(A)::SharedData A_sdata;
    __shared__ typename decltype(B)::SharedData B_sdata;
    __shared__ typename decltype(Cin)::SharedData Cin_sdata;
    __shared__ typename decltype(C)::SharedData C_sdata;
    for (int r = 0; r < batch_size; ++r) {
        A.init_async(A_sdata);
        B.init_async(B_sdata);
        Cin.init_async(Cin_sdata);
        C.init_async(C_sdata);
        if(decltype(A)::SYNC_AFTER_INIT   ||
        decltype(B)::SYNC_AFTER_INIT   ||
        decltype(Cin)::SYNC_AFTER_INIT ||
        decltype(C)::SYNC_AFTER_INIT    ) {
            __syncthreads();
        }
        A.init_async_postsync();
        B.init_async_postsync();
        Cin.init_async_postsync();
        C.init_async_postsync();

        if (ADD_TO_C) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = *Cin.align(oiter);
            }
        } else {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = Cinit;
            }
        }

        // Perform the multiplication
        for (int j = 0; j < A.chunks(); ++j) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                // Initialize B's iterator as well
                auto B_iter = B.align(oiter);

#pragma unroll
                MAPS_FOREACH_ALIGNED(A_iter, A, oiter) {
                    *oiter = ACC_F((*oiter), COMB_F((*A_iter), (*B_iter), extra_params));
                    ++B_iter;
                }
            }
            // Advance chunks efficiently
            maps::NextChunkAll(A, B);
        }

        // Write out results
        if (C.Items() > 0) {
            C.commit();
        }
        __syncthreads();
        if (BATCH_A_ACTIVE) {
            A.m_ptr = (void*)(((Atype*)A.m_ptr) + A_batch_stride);
        }
        if (BATCH_B_ACTIVE) {
            B.m_ptr = (void*)(((Btype*)B.m_ptr) + B_batch_stride);
        }
        if (BATCH_C_ACTIVE) {
            C.m_ptr = (void*)(((Ctype*)C.m_ptr) + C_batch_stride);
            Cin.m_ptr = (void*)(((Ctype*)Cin.m_ptr) + C_batch_stride);
        }
    }
}

template <typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_gpu(const int M, const int N, const int K,
               const Atype* A, const Btype* B, Ctype* C,
               const Atype Ainit, const Btype Binit, const Ctype Cinit,
               const Ptype extra_params, const int batch_size = 1,
               int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1,
               const cudaStream_t stream = 0) {
    if (BATCH_A_ACTIVE) {
        if (A_batch_stride < 0) {
            A_batch_stride = M * K;
        }
    }
    if (BATCH_B_ACTIVE) {
        if (B_batch_stride < 0) {
            B_batch_stride = N * K;
        }
    }
    if (BATCH_C_ACTIVE) {
        if (C_batch_stride < 0) {
            C_batch_stride = M * N;
        }
    }
    maps::BlockSingleGPU<Atype, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> A_maps;
    maps::BlockSingleGPU<Btype, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> B_maps;
    maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries, ILP_X, ILP_Y, 1, maps::DirectIO> Cin_maps;
    maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C_maps;

    A_maps.m_ptr = (void*) A;
    A_maps.m_dimensions[0] = K;
    A_maps.m_dimensions[1] = M;
    A_maps.m_stride = A_maps.m_dimensions[0];
    A_maps.m_init_value = Ainit;

    B_maps.m_ptr = (void*) B;
    B_maps.m_dimensions[0] = N;
    B_maps.m_dimensions[1] = K;
    B_maps.m_stride = B_maps.m_dimensions[0];
    B_maps.m_init_value = Binit;

    Cin_maps.m_ptr = C;
    Cin_maps.m_dimensions[0] = N;
    Cin_maps.m_dimensions[1] = M;
    Cin_maps.m_stride = Cin_maps.m_dimensions[0];

    C_maps.m_ptr = C;
    C_maps.m_dimensions[0] = N;
    C_maps.m_dimensions[1] = M;
    C_maps.m_stride = C_maps.m_dimensions[0];

    dim3 block_dims (BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid_dims (maps::RoundUp(N, block_dims.x), maps::RoundUp(M, block_dims.y));

    ggemm_kernel<BLOCK_WIDTH, BLOCK_HEIGHT, ILP_X, ILP_Y,
            Atype, Btype, Ctype, Ptype,
            COMB_F, ACC_F, ADD_TO_C,
            BATCH_A_ACTIVE, BATCH_B_ACTIVE, BATCH_C_ACTIVE>
            <<<grid_dims, block_dims, 0, stream>>>
                                         (A_maps, B_maps, Cin_maps, C_maps, Cinit, extra_params, batch_size,
                                                 A_batch_stride, B_batch_stride, C_batch_stride);
}

template <int BLOCK_WIDTH, int BLOCK_HEIGHT, int ILP_X, int ILP_Y,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F1)(Atype, Btype, Ptype), Ctype (*ACC_F1)(Ctype, Ctype),
        Ctype (*COMB_F2)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F2)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
__global__ void ggemm_2ops_kernel(
        maps::BlockSingleGPU<Atype, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1, maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> A,
        maps::BlockSingleGPU<Btype, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1, maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> B,
        maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries, ILP_X, ILP_Y, 1, maps::DirectIO> Cin,
        maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> Ctemp,
        maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C,
        const Ctype Cinit1, const Ctype Cinit2, const Ptype extra_params, const int batch_size = 1,
        const int A_batch_stride = -1, const int B_batch_stride = -1, const int C_batch_stride = -1) {
    __shared__ typename decltype(A)::SharedData A_sdata;
    __shared__ typename decltype(B)::SharedData B_sdata;
    __shared__ typename decltype(Cin)::SharedData Cin_sdata;
    __shared__ typename decltype(C)::SharedData C_sdata;
    __shared__ typename decltype(Ctemp)::SharedData Ctemp_sdata;
    for (int r = 0; r < batch_size; ++r) {
        // Initialize data
        A.init_async(A_sdata);
        B.init_async(B_sdata);
        Ctemp.init_async(Ctemp_sdata);
        if(decltype(A)::SYNC_AFTER_INIT     ||
        decltype(B)::SYNC_AFTER_INIT     ||
        decltype(Ctemp)::SYNC_AFTER_INIT  ) {
            __syncthreads();
        }
        A.init_async_postsync();
        B.init_async_postsync();
        Ctemp.init_async_postsync();

#pragma unroll
        MAPS_FOREACH(oiter, Ctemp) {
            *oiter = Cinit1;
        }

        // Perform the multiplication
        for (int j = 0; j < A.chunks(); ++j) {
#pragma unroll
            MAPS_FOREACH(oiter, Ctemp) {
                // Initialize B's iterator as well
                auto B_iter = B.align(oiter);

#pragma unroll
                MAPS_FOREACH_ALIGNED(A_iter, A, oiter) {
                    *oiter = ACC_F1((*oiter), COMB_F1((*A_iter), (*B_iter), extra_params));
                    ++B_iter;
                }
            }
            // Advance chunks efficiently
            maps::NextChunkAll(A, B);
        }
        A.init_async(A_sdata);
        B.init_async(B_sdata);
        Cin.init_async(Cin_sdata);
        C.init_async(C_sdata);
        if(decltype(A)::SYNC_AFTER_INIT   ||
        decltype(B)::SYNC_AFTER_INIT   ||
        decltype(Cin)::SYNC_AFTER_INIT ||
        decltype(C)::SYNC_AFTER_INIT    ) {
            __syncthreads();
        }
        A.init_async_postsync();
        B.init_async_postsync();
        Cin.init_async_postsync();
        C.init_async_postsync();


        if (ADD_TO_C && !APPLY_ON_C) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = *Cin.align(oiter);
            }
        } else {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = Cinit2;
            }
        }

        // Perform the multiplication
        for (int j = 0; j < A.chunks(); ++j) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                // Initialize B's iterator as well
                auto B_iter = B.align(oiter);
                auto Ctemp_iter = Ctemp.begin();
#pragma unroll
                MAPS_FOREACH_ALIGNED(A_iter, A, oiter) {
                    *oiter = ACC_F2((*oiter), COMB_F2((*A_iter), (*B_iter), (*Ctemp_iter), extra_params));
                    ++B_iter;
                }
                ++Ctemp_iter;
            }
            // Advance chunks efficiently
            maps::NextChunkAll(A, B);
        }

        if (APPLY_ON_C) {
            if (ADD_TO_C) {
#pragma unroll
                MAPS_FOREACH(oiter, C) {
                    auto Ctemp_iter = Ctemp.begin();
                    *oiter = ACC_F2(*Cin.align(oiter), APPLY_F(*Ctemp_iter, *oiter, extra_params));
                    ++Ctemp_iter;
                }
            } else {
#pragma unroll
                MAPS_FOREACH(oiter, C) {
                    auto Ctemp_iter = Ctemp.begin();
                    *oiter = APPLY_F(*Ctemp_iter, *oiter, extra_params);
                    ++Ctemp_iter;
                }
            }
        }

        // Write out results
        if (C.Items() > 0) {
            C.commit();
        }
        __syncthreads();
        if (BATCH_A_ACTIVE) {
            A.m_ptr = (void*)(((Atype*)A.m_ptr) + A_batch_stride);
        }
        if (BATCH_B_ACTIVE) {
            B.m_ptr = (void*)(((Btype*)B.m_ptr) + B_batch_stride);
        }
        if (BATCH_C_ACTIVE) {
            C.m_ptr = (void*)(((Ctype*)C.m_ptr) + C_batch_stride);
            Cin.m_ptr = (void*)(((Ctype*)Cin.m_ptr) + C_batch_stride);
            Ctemp.m_ptr = (void*)(((Ctype*)Ctemp.m_ptr) + C_batch_stride);
        }
    }
}

template <typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F1)(Atype, Btype, Ptype), Ctype (*ACC_F1)(Ctype, Ctype),
        Ctype (*COMB_F2)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F2)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_2ops_gpu(const int M, const int N, const int K,
                    const Atype* A, const Btype* B, Ctype* C,
                    const Atype Ainit, const Btype Binit, const Ctype Cinit1, const Ctype Cinit2,
                    const Ptype extra_params, const int batch_size = 1,
                    int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1,
                    const cudaStream_t stream = 0) {
    if (BATCH_A_ACTIVE) {
        if (A_batch_stride < 0) {
            A_batch_stride = M * K;
        }
    }
    if (BATCH_B_ACTIVE) {
        if (B_batch_stride < 0) {
            B_batch_stride = N * K;
        }
    }
    if (BATCH_C_ACTIVE) {
        if (C_batch_stride < 0) {
            C_batch_stride = M * N;
        }
    }
    maps::BlockSingleGPU<Atype, 2, 0, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> A_maps;
    maps::BlockSingleGPU<Btype, 2, 1, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries, BLOCK_WIDTH, BLOCK_HEIGHT, 1> B_maps;
    maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries, ILP_X, ILP_Y, 1, maps::DirectIO> Cin_maps;
    maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C_maps;
    maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> Ctemp;

    A_maps.m_ptr = (void*) A;
    A_maps.m_dimensions[0] = K;
    A_maps.m_dimensions[1] = M;
    A_maps.m_stride = A_maps.m_dimensions[0];
    A_maps.m_init_value = Ainit;

    B_maps.m_ptr = (void*) B;
    B_maps.m_dimensions[0] = N;
    B_maps.m_dimensions[1] = K;
    B_maps.m_stride = B_maps.m_dimensions[0];
    B_maps.m_init_value = Binit;

    Cin_maps.m_ptr = C;
    Cin_maps.m_dimensions[0] = N;
    Cin_maps.m_dimensions[1] = M;
    Cin_maps.m_stride = Cin_maps.m_dimensions[0];

    C_maps.m_ptr = C;
    C_maps.m_dimensions[0] = N;
    C_maps.m_dimensions[1] = M;
    C_maps.m_stride = C_maps.m_dimensions[0];

    Ctemp.m_ptr = C_maps.m_ptr;
    Ctemp.m_dimensions[0] = C_maps.m_dimensions[0];
    Ctemp.m_dimensions[1] = C_maps.m_dimensions[1];
    Ctemp.m_stride = C_maps.m_stride;

    dim3 block_dims (BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid_dims (maps::RoundUp(N, block_dims.x), maps::RoundUp(M, block_dims.y));

    ggemm_2ops_kernel<BLOCK_WIDTH, BLOCK_HEIGHT, ILP_X, ILP_Y,
            Atype, Btype, Ctype, Ptype,
            COMB_F1, ACC_F1, COMB_F2, ACC_F2, ADD_TO_C, APPLY_F, APPLY_ON_C,
            BATCH_A_ACTIVE, BATCH_B_ACTIVE, BATCH_C_ACTIVE>
            <<<grid_dims, block_dims, 0, stream>>>
                                         (A_maps, B_maps, Cin_maps, Ctemp, C_maps, Cinit1, Cinit2, extra_params, batch_size,
                                                 A_batch_stride, B_batch_stride, C_batch_stride);
}


template <bool TRANSPOSE_A, bool TRANSPOSE_B, int BLOCK_WIDTH, int BLOCK_HEIGHT, int ILP_X, int ILP_Y,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
__global__ void ggemm_readc_kernel(
        maps::BlockSingleGPU<Atype, 2, TRANSPOSE_A ? 1 : 0,
                BLOCK_WIDTH, BLOCK_HEIGHT, 1,
                ILP_X, ILP_Y, 1,
                maps::ZeroBoundaries,
                BLOCK_WIDTH, BLOCK_HEIGHT, 1,
                maps::CustomOrdering<(TRANSPOSE_A ? 1 : 0), (TRANSPOSE_A ? 0 : 1)> > A,
        maps::BlockSingleGPU<Btype, 2, TRANSPOSE_B ? 0 : 1,
                BLOCK_WIDTH, BLOCK_HEIGHT, 1,
                ILP_X, ILP_Y, 1,
                maps::ZeroBoundaries,
                BLOCK_WIDTH, BLOCK_HEIGHT, 1,
                maps::CustomOrdering<(TRANSPOSE_B ? 1 : 0), (TRANSPOSE_B ? 0 : 1)> > B,
        maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries,
                ILP_X, ILP_Y, 1, maps::DirectIO> Cin,
        maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries,
                ILP_X, ILP_Y, 1, maps::DirectIO> Cinout,
        maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C,
        const Ctype Cinit, const Ptype extra_params, const int batch_size = 1,
        const int A_batch_stride = -1, const int B_batch_stride = -1, const int C_batch_stride = -1) {

    __shared__ typename decltype(A)::SharedData A_sdata;
    __shared__ typename decltype(B)::SharedData B_sdata;
    __shared__ typename decltype(Cin)::SharedData Cin_sdata;
    __shared__ typename decltype(Cinout)::SharedData Cinout_sdata;
    __shared__ typename decltype(C)::SharedData C_sdata;
    for (int r = 0; r < batch_size; ++r) {
        A.init_async(A_sdata);
        B.init_async(B_sdata);
        Cin.init_async(Cin_sdata);
        Cinout.init_async(Cinout_sdata);
        C.init_async(C_sdata);
        if(decltype(A)::SYNC_AFTER_INIT      ||
        decltype(B)::SYNC_AFTER_INIT      ||
        decltype(Cin)::SYNC_AFTER_INIT    ||
        decltype(Cinout)::SYNC_AFTER_INIT ||
        decltype(C)::SYNC_AFTER_INIT)      {
            __syncthreads();
        }
        A.init_async_postsync();
        B.init_async_postsync();
        Cin.init_async_postsync();
        Cinout.init_async_postsync();
        C.init_async_postsync();

        if (ADD_TO_C && !APPLY_ON_C) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = *Cinout.align(oiter);
            }
        } else {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                *oiter = Cinit;
            }
        }

        // Perform the multiplication
        for (int j = 0; j < A.chunks(); ++j) {
#pragma unroll
            MAPS_FOREACH(oiter, C) {
                // Initialize B's iterator as well
                auto B_iter = B.align(oiter);
                Ctype c_orig = *Cin.align(oiter);
#pragma unroll
                MAPS_FOREACH_ALIGNED(A_iter, A, oiter) {
                    *oiter = ACC_F((*oiter), COMB_F((*A_iter), (*B_iter), c_orig, extra_params));
                    ++B_iter;
                }
            }
            // Advance chunks efficiently
            maps::NextChunkAll(A, B);
        }

        if (APPLY_ON_C) {
            if (ADD_TO_C) {
#pragma unroll
                MAPS_FOREACH(oiter, C) {
                    *oiter = ACC_F(*Cinout.align(oiter), APPLY_F(*oiter, extra_params));
                }
            } else {
#pragma unroll
                MAPS_FOREACH(oiter, C) {
                    *oiter = APPLY_F(*oiter, extra_params);
                }
            }
        }
        // Write out results
        if (C.Items() > 0) {
            C.commit();
        }
        __syncthreads();
        if (BATCH_A_ACTIVE) {
            A.m_ptr = (void*)(((Atype*)A.m_ptr) + A_batch_stride);
        }
        if (BATCH_B_ACTIVE) {
            B.m_ptr = (void*)(((Btype*)B.m_ptr) + B_batch_stride);
        }
        if (BATCH_C_ACTIVE) {
            C.m_ptr = (void*)(((Ctype*)C.m_ptr) + C_batch_stride);
            Cin.m_ptr = (void*)(((Ctype*)Cin.m_ptr) + C_batch_stride);
            Cinout.m_ptr = (void*)(((Ctype*)Cinout.m_ptr) + C_batch_stride);
        }
    }
}


template <bool TRANSPOSE_A, bool TRANSPOSE_B,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_readc_gpu(const int M, const int N, const int K,
                     const Atype* A, const Btype* B, const Ctype* Cin, Ctype* Cout,
                     const Atype Ainit, const Btype Binit, const Ctype Cinit,
                     const Ptype extra_params, const int batch_size = 1,
                     int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1,
                     const cudaStream_t stream = 0) {

    if (BATCH_A_ACTIVE) {
        if (A_batch_stride < 0) {
            A_batch_stride = M * K;
        }
    }
    if (BATCH_B_ACTIVE) {
        if (B_batch_stride < 0) {
            B_batch_stride = N * K;
        }
    }
    if (BATCH_C_ACTIVE) {
        if (C_batch_stride < 0) {
            C_batch_stride = M * N;
        }
    }

    maps::BlockSingleGPU<Atype, 2, TRANSPOSE_A ? 1 : 0,
            BLOCK_WIDTH, BLOCK_HEIGHT, 1,
            ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries,
            BLOCK_WIDTH, BLOCK_HEIGHT, 1,
            maps::CustomOrdering<(TRANSPOSE_A ? 1 : 0), (TRANSPOSE_A ? 0 : 1)> > A_maps;
    maps::BlockSingleGPU<Btype, 2, TRANSPOSE_B ? 0 : 1,
            BLOCK_WIDTH, BLOCK_HEIGHT, 1,
            ILP_X, ILP_Y, 1,
            maps::ZeroBoundaries,
            BLOCK_WIDTH, BLOCK_HEIGHT, 1,
            maps::CustomOrdering<(TRANSPOSE_B ? 1 : 0), (TRANSPOSE_B ? 0 : 1)> > B_maps;
    maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries,
            ILP_X, ILP_Y, 1, maps::DirectIO> Cin_maps;
    maps::Window2DSingleGPU<Ctype, BLOCK_WIDTH, BLOCK_HEIGHT, 0, maps::NoBoundaries,
            ILP_X, ILP_Y, 1, maps::DirectIO> Cinout_maps;
    maps::StructuredInjectiveSingleGPU<Ctype, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, ILP_X, ILP_Y> C_maps;

    A_maps.m_ptr = (void*) A;
    A_maps.m_dimensions[0] = TRANSPOSE_A ? M : K;
    A_maps.m_dimensions[1] = TRANSPOSE_A ? K : M;
    A_maps.m_stride = A_maps.m_dimensions[0];
    A_maps.m_init_value = Ainit;

    B_maps.m_ptr = (void*) B;
    B_maps.m_dimensions[0] = TRANSPOSE_B ? K : N;
    B_maps.m_dimensions[1] = TRANSPOSE_B ? N : K;
    B_maps.m_stride = B_maps.m_dimensions[0];
    B_maps.m_init_value = Binit;

    Cin_maps.m_ptr = (void*) Cin;
    Cin_maps.m_dimensions[0] = N;
    Cin_maps.m_dimensions[1] = M;
    Cin_maps.m_stride = Cin_maps.m_dimensions[0];

    Cinout_maps.m_ptr = Cout;
    Cinout_maps.m_dimensions[0] = N;
    Cinout_maps.m_dimensions[1] = M;
    Cinout_maps.m_stride = Cin_maps.m_dimensions[0];

    C_maps.m_ptr = Cout;
    C_maps.m_dimensions[0] = N;
    C_maps.m_dimensions[1] = M;
    C_maps.m_stride = C_maps.m_dimensions[0];

    dim3 block_dims (BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid_dims (maps::RoundUp(N, block_dims.x), maps::RoundUp(M, block_dims.y));

    ggemm_readc_kernel<TRANSPOSE_A, TRANSPOSE_B, BLOCK_WIDTH, BLOCK_HEIGHT, ILP_X, ILP_Y,
            Atype, Btype, Ctype, Ptype,
            COMB_F, ACC_F, ADD_TO_C, APPLY_F, APPLY_ON_C,
            BATCH_A_ACTIVE, BATCH_B_ACTIVE, BATCH_C_ACTIVE>
            <<<grid_dims, block_dims, 0, stream>>>
                                         (A_maps, B_maps, Cin_maps, Cinout_maps, C_maps, Cinit, extra_params, batch_size,
                                                 A_batch_stride, B_batch_stride, C_batch_stride);
}

template <typename Dtype, bool REVERSE>
__global__ void split_patches_kernel(const int num_kernels, const int N, const int Dim,
                                     const int W, const int H, const int C,
                                     const int W_Gs, const int H_Gs, const int C_Gs,
                                     const int W_Step, const int H_Step, const int C_Step,
                                     typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
                                     Dtype* out, const bool use_unshared_regions_) {
    const int step_out = C_Step * H_Step * W_Step;
    const int group_step_w = !use_unshared_regions_ ? W_Step : 1;
    const int group_step_h = !use_unshared_regions_ ? H_Step : 1;
    const int group_step_c = !use_unshared_regions_ ? C_Step : 1;
    const int region_step_w = !use_unshared_regions_ ? 1 : W_Gs;
    const int region_step_h = !use_unshared_regions_ ? 1 : H_Gs;
    const int region_step_c = !use_unshared_regions_ ? 1 : C_Gs;
    Dtype* in_unconst = NULL;
    if (REVERSE) {
        in_unconst = (Dtype*)in;
    }
    CUDA_KERNEL_LOOP(index, num_kernels) {
        const int i = index % W_Step;
        const int i_index = index / W_Step;
        const int j = i_index % H_Step;
        const int j_index = i_index / H_Step;
        const int l = j_index % C_Step;
        const int l_index = j_index / C_Step;
        const int w_g = l_index % W_Gs;
        const int w_index = l_index / W_Gs;
        const int h_g = w_index % H_Gs;
        const int h_index = w_index / H_Gs;
        const int c_g = h_index;

        // "inner loop"
        Dtype* o = out + ((c_g * H_Gs + h_g) * W_Gs + w_g) * step_out * Dim;
        const int group_addr = (c_g * group_step_c * H + h_g * group_step_h) * W + w_g * group_step_w;
        const int base_addr_out = (l * H_Step + j) * W_Step + i;
        const int base_addr_in  = group_addr + (l * region_step_c * H + j * region_step_h) * W  + i * region_step_w;
        if (w_g * group_step_w + i * region_step_w < W &&
            h_g * group_step_h + j * region_step_h < H &&
            c_g * group_step_c + l * region_step_c < C) {
            for (int k = 0; k < Dim; ++k) {
                if (!REVERSE) {
                    o[base_addr_out + k * step_out] = in[base_addr_in + k * N];
                } else {
                    in_unconst[base_addr_in + k * N] = o[base_addr_out + k * step_out];
                }
            }
        }
    }
}


template <typename Dtype, bool REVERSE>
inline
void split_patches_gpu(const int N, const int Dim,
                       const int W, const int H, const int C,
                       const int W_Gs, const int H_Gs, const int C_Gs,
                       const int W_Step, const int H_Step, const int C_Step,
                       typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
                       Dtype* out, const bool use_unshared_regions) {
    const int num_kernels = W_Step * H_Step * C_Step * W_Gs * H_Gs * C_Gs;
    // NOLINT_NEXT_LINE(whitespace/operators)
    split_patches_kernel<Dtype, REVERSE><<<GGEMM_GET_BLOCKS(num_kernels), GGEMM_CUDA_NUM_THREADS>>>(
            num_kernels, N, Dim, W, H, C, W_Gs, H_Gs, C_Gs, W_Step, H_Step, C_Step, in, out, use_unshared_regions);
    CUDA_POST_KERNEL_CHECK;
}

#endif