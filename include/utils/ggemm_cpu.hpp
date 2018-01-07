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

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
#endif

#ifndef _GGEMM_CPU_H_
#define _GGEMM_CPU_H_
#include <cmath>
#ifdef SIMNETS_WITH_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#else
    #define __host__
    #define __device__
    #define __forceinline__

#define VEC_TYPES(typ) \
struct typ##2 {typ x, y;}; \
struct typ##3 {typ x, y, z;}; \
struct typ##4 {typ x, y, z, w;};

VEC_TYPES(int)
VEC_TYPES(long)
VEC_TYPES(float)
VEC_TYPES(double)

#endif
#include <type_traits>

// Templated vectorized types
struct intvec2 : public int2 {
    intvec2() = default;
    __device__ __host__  __forceinline__ intvec2(int in) {
        this->x = in;
        this->y = in;
    }
    __device__ __host__  __forceinline__ intvec2(int in1, int in2) {
        this->x = in1;
        this->y = in2;
    }
    __device__ __host__  __forceinline__ intvec2(int2 in) {
        this->x = in.x;
        this->y = in.y;
    }
    __device__ __host__  __forceinline__ intvec2 & operator= (int2 in) {
        this->x = in.x;
        this->y = in.y;
        return *this;
    }
};
struct intvec3 : public int3 {
    intvec3() = default;
    __device__ __host__  __forceinline__ intvec3(int in) {
        this->x = in;
        this->y = in;
        this->z = in;
    }
    __device__ __host__  __forceinline__ intvec3(int in1, int in2, int in3) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
    }
    __device__ __host__  __forceinline__ intvec3(int3 in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
    }
    __device__ __host__  __forceinline__ intvec3 & operator=(int3 in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        return *this;
    }
};
struct intvec4 : public int4 {
    intvec4() = default;
    __device__ __host__  __forceinline__ intvec4(int in) {
        this->x = in;
        this->y = in;
        this->z = in;
        this->w = in;
    }
    __device__ __host__  __forceinline__ intvec4(int in1, int in2, int in3, int in4) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
        this->w = in4;
    }
    __device__ __host__  __forceinline__ intvec4(int4 in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
    }
    __device__ __host__  __forceinline__ intvec4 & operator=(int4 in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
        return *this;
    }
};
struct longvec2 : public long2 {
    longvec2() = default;
    __device__ __host__  __forceinline__ longvec2(long in) {
        this->x = in;
        this->y = in;
    }
    __device__ __host__  __forceinline__ longvec2(long in1, long in2) {
        this->x = in1;
        this->y = in2;
    }
    __device__ __host__  __forceinline__ longvec2(long2 in) {
        this->x = in.x;
        this->y = in.y;
    }
    __device__ __host__  __forceinline__ longvec2 & operator=(long2 in) {
        this->x = in.x;
        this->y = in.y;
        return *this;
    }
};
struct longvec3 : public long3 {
    longvec3() = default;
    __device__ __host__  __forceinline__ longvec3(long in) {
        this->x = in;
        this->y = in;
        this->z = in;
    }
    __device__ __host__  __forceinline__ longvec3(long in1, long in2, long in3) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
    }
    __device__ __host__  __forceinline__ longvec3(const float3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
    }
    __device__ __host__  __forceinline__ longvec3 & operator=(const long3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        return *this;
    }
};
struct longvec4 : public long4 {
    longvec4() = default;
    __device__ __host__  __forceinline__ longvec4(long in) {
        this->x = in;
        this->y = in;
        this->z = in;
        this->w = in;
    }
    __device__ __host__  __forceinline__ longvec4(long in1, long in2, long in3, long in4) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
        this->w = in4;
    }
    __device__ __host__  __forceinline__ longvec4(const long4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
    }
    __device__ __host__  __forceinline__ longvec4 & operator=(const long4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
        return *this;
    }
};
struct floatvec2 : public float2 {
    floatvec2() = default;
    __device__ __host__  __forceinline__ floatvec2(float in) {
        this->x = in;
        this->y = in;
    }
    __device__ __host__  __forceinline__ floatvec2(float in1, float in2) {
        this->x = in1;
        this->y = in2;
    }
    __device__ __host__  __forceinline__ floatvec2(const float2 &in) {
        this->x = in.x;
        this->y = in.y;
    }
    __device__ __host__  __forceinline__ floatvec2 & operator=(const float2 &in) {
        this->x = in.x;
        this->y = in.y;
        return *this;
    }
};
struct floatvec3 : public float3 {
    floatvec3() = default;
    __device__ __host__  __forceinline__ floatvec3(float in) {
        this->x = in;
        this->y = in;
        this->z = in;
    }
    __device__ __host__  __forceinline__ floatvec3(float in1, float in2, float in3) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
    }
    __device__ __host__  __forceinline__ floatvec3(const float3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
    }
    __device__ __host__  __forceinline__ floatvec3 & operator=(const float3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        return *this;
    }
};
struct floatvec4 : public float4 {
    floatvec4() = default;
    __device__ __host__  __forceinline__ floatvec4(float in) {
        this->x = in;
        this->y = in;
        this->z = in;
        this->w = in;
    }
    __device__ __host__  __forceinline__ floatvec4(float in1, float in2, float in3, float in4) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
        this->w = in4;
    }
    __device__ __host__  __forceinline__ floatvec4(const float4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
    }
    __device__ __host__  __forceinline__ floatvec4 & operator=(const float4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
        return *this;
    }
};
struct doublevec2 : public double2 {
    doublevec2() = default;
    __device__ __host__  __forceinline__ doublevec2(double in) {
        this->x = in;
        this->y = in;
    }
    __device__ __host__  __forceinline__ doublevec2(double in1, double in2) {
        this->x = in1;
        this->y = in2;
    }
    __device__ __host__  __forceinline__ doublevec2(const double2 &in) {
        this->x = in.x;
        this->y = in.y;
    }
    __device__ __host__  __forceinline__ doublevec2 & operator=(const double2 &in) {
        this->x = in.x;
        this->y = in.y;
        return *this;
    }
};
struct doublevec3 {
    double x, y, z;
    doublevec3() = default;
    __device__ __host__  __forceinline__ doublevec3(double in) {
        this->x = in;
        this->y = in;
        this->z = in;
    }
    __device__ __host__  __forceinline__ doublevec3(double in1, double in2, double in3) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
    }
    __device__ __host__  __forceinline__ doublevec3(const double3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
    }
    __device__ __host__  __forceinline__ doublevec3 & operator=(const double3 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        return *this;
    }
};

struct doublevec4 {
    double x, y, z, w;
    doublevec4() = default;
    __device__ __host__  __forceinline__ doublevec4(double in) {
        this->x = in;
        this->y = in;
        this->z = in;
        this->w = in;
    }
    __device__ __host__  __forceinline__ doublevec4(double in1, double in2, double in3, double in4) {
        this->x = in1;
        this->y = in2;
        this->z = in3;
        this->w = in4;
    }
    __device__ __host__  __forceinline__ doublevec4(const double4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
    }
    __device__ __host__  __forceinline__ doublevec4 & operator=(const double4 &in) {
        this->x = in.x;
        this->y = in.y;
        this->z = in.z;
        this->w = in.w;
        return *this;
    }
};

template<typename T> struct vec;
template<> struct vec<int>    {typedef int vec1; typedef intvec2 vec2; typedef intvec3 vec3; typedef intvec4 vec4;};
template<> struct vec<long>    {typedef long vec1; typedef longvec2 vec2; typedef longvec3 vec3; typedef longvec4 vec4;};
template<> struct vec<float>  {typedef float vec1; typedef floatvec2 vec2; typedef floatvec3 vec3; typedef floatvec4 vec4;};
template<> struct vec<double> {
    typedef double vec1;
    typedef doublevec2 vec2;
    typedef doublevec3 vec3;
    typedef doublevec4 vec4;
};

// Templated constructors
template<typename T> __device__ __host__ __forceinline__ typename vec<T>::vec1 make_vec1(T x) {
    return x;
}
template<typename T> __device__ __host__ __forceinline__ typename vec<T>::vec2 make_vec2(T x, T y) {
    return typename vec<T>::vec2(x, y);
}
template<typename T> __device__ __host__ __forceinline__ typename vec<T>::vec3 make_vec3(T x, T y, T z) {
    return typename vec<T>::vec3(x, y, z);
}
template<typename T> __device__ __host__ __forceinline__ typename vec<T>::vec4 make_vec4(T x, T y, T z, T w) {
    return typename vec<T>::vec4(x, y, z, w);
}


template<typename Dtype, typename Ptype> __device__ __host__ __forceinline__
Dtype no_op(Dtype a, Ptype nothing) {
    return a;
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype ggemm_max(Dtype a, Dtype b) {
#ifdef __CUDA_ARCH__
    return max(a,b);
#else
    return std::max(a, b);
#endif
}

template<typename Dtype, bool INTRINSIC = false> __device__ __host__ __forceinline__
Dtype ggemm_log(Dtype a) {
#ifdef __CUDA_ARCH__
    if (INTRINSIC) {
    return __logf(a);
  } else {
    return log(a);
  }
#else
    return std::log(a);
#endif
}
template<typename Dtype, typename Ptype, bool INTRINSIC = false> __device__ __host__ __forceinline__
Dtype ggemm_log(Dtype a, Ptype param) {
    return ggemm_log<Dtype, INTRINSIC>(a);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype ggemm_min(Dtype a, Dtype b) {
#ifdef __CUDA_ARCH__
    return min(a,b);
#else
    return std::min(a, b);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype ggemm_mul(Dtype a, Dtype b) {
    return a * b;
}

template<typename Dtype, typename Ntype> __device__ __host__ __forceinline__
Dtype ggemm_add(Dtype a, Dtype b, Ntype nothing) {
    return a + b;
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype ggemm_add(Dtype a, Dtype b) {
    return a + b;
}

template<typename Dtype> __device__ __host__ __forceinline__
typename vec<Dtype>::vec2 add_vec2(typename vec<Dtype>::vec2 a, typename vec<Dtype>::vec2 b) {
    return make_vec2<Dtype>(a.x + b.x, a.y + b.y);
}

template<typename Dtype> __device__ __host__ __forceinline__
typename vec<Dtype>::vec4 add_vec4(typename vec<Dtype>::vec4 a, typename vec<Dtype>::vec4 b) {
    return make_vec4<Dtype>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <typename Dtype> void interlace_cpu(const int N, const Dtype* a, const Dtype* b, typename vec<Dtype>::vec2* o) {
    for (int i = 0; i < N; ++i)
    {
        o[i] = make_vec2<Dtype>(a[i], b[i]);
    }
}

template <typename Dtype> void deinterlace_cpu(const int N,
                                               const typename vec<Dtype>::vec2* in, Dtype* a, Dtype* b) {
    for (int i = 0; i < N; ++i)
    {
        const typename vec<Dtype>::vec2 data = in[i];
        a[i] = data.x;
        b[i] = data.y;
    }
}

enum {
    ILP_X = 1,
    ILP_Y = 1,
    BLOCK_WIDTH = 16,
    BLOCK_HEIGHT = 16
};

template <typename Dtype>
inline Dtype ceiled_div(const Dtype a, const Dtype b) {
    return (a / b) + ((a % b) > 0);
}

inline int ggemm_padded_output_size(const int M, const int N) {
    int newN = ceiled_div<int>(N, BLOCK_WIDTH) * BLOCK_WIDTH;
    int newM = ceiled_div<int>(M, BLOCK_HEIGHT) * BLOCK_HEIGHT;
    return newN * newM - M * N;
}

template <typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_cpu(const int M, const int N, const int K,
               const Atype* A, const Btype* B, Ctype* C,
               const Ctype Cinit, const Ptype extra_params, const int batch_size = 1,
               int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1) {
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
    for (int r = 0; r < batch_size; ++r) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Ctype sum = Cinit;
                for (int k = 0; k < K; ++k) {
                    Atype a = A[i * K + k];
                    Btype b = B[k * N + j];
                    Ctype temp = COMB_F(a, b, extra_params);
                    sum = ACC_F(sum, temp);
                }
                if (ADD_TO_C) {
                    C[i * N + j] = ACC_F(C[i * N + j], sum);
                } else {
                    C[i * N + j] = sum;
                }
            }
        }
        if (BATCH_A_ACTIVE) {
            A += A_batch_stride;
        }
        if (BATCH_B_ACTIVE) {
            B += B_batch_stride;
        }
        if (BATCH_C_ACTIVE) {
            C += C_batch_stride;
        }
    }
}

template <typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F1)(Atype, Btype, Ptype), Ctype (*ACC_F1)(Ctype, Ctype),
        Ctype (*COMB_F2)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F2)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_2ops_cpu(const int M, const int N, const int K,
                    const Atype* A, const Btype* B, Ctype* C,
                    const Ctype Cinit1, const Ctype Cinit2, const Ptype extra_params, const int batch_size = 1,
                    int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1) {
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
    for (int r = 0; r < batch_size; ++r) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Ctype sum1 = Cinit1;
                for (int k = 0; k < K; ++k) {
                    Atype a = A[i * K + k];
                    Btype b = B[k * N + j];
                    Ctype temp = COMB_F1(a, b, extra_params);
                    sum1 = ACC_F1(sum1, temp);
                }
                Ctype sum2 = Cinit2;
                for (int k = 0; k < K; ++k) {
                    Atype a = A[i * K + k];
                    Btype b = B[k * N + j];
                    Ctype temp = COMB_F2(a, b, sum1, extra_params);
                    sum2 = ACC_F2(sum2, temp);
                }
                Ctype final_value;
                if (APPLY_ON_C) {
                    final_value = APPLY_F(sum1, sum2, extra_params);
                } else {
                    final_value = sum2;
                }
                if (ADD_TO_C) {
                    C[i * N + j] = ACC_F2(C[i * N + j], final_value);
                } else {
                    C[i * N + j] = final_value;
                }
            }
        }
        if (BATCH_A_ACTIVE) {
            A += A_batch_stride;
        }
        if (BATCH_B_ACTIVE) {
            B += B_batch_stride;
        }
        if (BATCH_C_ACTIVE) {
            C += C_batch_stride;
        }
    }
}

template <bool TRANSPOSE_A, bool TRANSPOSE_B,
        typename Atype, typename Btype, typename Ctype, typename Ptype,
        Ctype (*COMB_F)(Atype, Btype, Ctype, Ptype), Ctype (*ACC_F)(Ctype, Ctype), bool ADD_TO_C,
        Ctype (*APPLY_F)(Ctype, Ptype), bool APPLY_ON_C,
        bool BATCH_A_ACTIVE = false, bool BATCH_B_ACTIVE = false, bool BATCH_C_ACTIVE = false>
void ggemm_readc_cpu(const int M, const int N, const int K,
                     const Atype* A, const Btype* B, const Ctype* Cin, Ctype* Cout,
                     const Ctype Cinit, const Ptype extra_params, const int batch_size = 1,
                     int A_batch_stride = -1, int B_batch_stride = -1, int C_batch_stride = -1) {
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
    for (int r = 0; r < batch_size; ++r) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                Ctype sum = Cinit;
                Ctype c = Cin[i * N + j];
                for (int k = 0; k < K; ++k) {
                    Atype a = A[TRANSPOSE_A ? k * M + i : i * K + k];
                    Btype b = B[TRANSPOSE_B ? j * K + k : k * N + j];
                    sum = ACC_F(sum, COMB_F(a, b, c, extra_params));
                }
                if (APPLY_ON_C) {
                    sum = APPLY_F(sum, extra_params);
                }
                if (ADD_TO_C) {
                    Cout[i * N + j] = ACC_F(Cout[i * N + j], sum);
                } else {
                    Cout[i * N + j] = sum;
                }
            }
        }
        if (BATCH_A_ACTIVE) {
            A += A_batch_stride;
        }
        if (BATCH_B_ACTIVE) {
            B += B_batch_stride;
        }
        if (BATCH_C_ACTIVE) {
            Cin += C_batch_stride;
            Cout += C_batch_stride;
        }
    }
}

#define ISNAN(x) (x != x)

template <typename T> __forceinline__ __device__ __host__ int sign(T val) {
    return (T(0) < val) - (val < T(0));
}


#endif
