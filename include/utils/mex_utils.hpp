//
// Created by elhanani on 26/07/17.
//

#ifndef SIMNETS_TF_MEX_UTILS_HPP
#define SIMNETS_TF_MEX_UTILS_HPP

#include "utils/ggemm_cpu.hpp"

#define EXP_CUDA __expf
template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_forward_exp(Dtype offset, Dtype data, Dtype max, typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
    return EXP_CUDA(extra.x * (data + offset - max) + extra.y);
  //return __fdividef(exp(data + offset - max), Dtype(K));
#else
    return std::exp(extra.x * (data + offset - max) + extra.y);
#endif
}
#define LOG_CUDA __logf

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_forward_out(Dtype in, typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
    return LOG_CUDA(in) / extra.x;
#else
    return std::log(in) / extra.x;
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_bottom_finite(Dtype offset, typename vec<Dtype>::vec2 top_data, Dtype data,
                                 typename vec<Dtype>::vec2 extra) {
#ifdef __CUDA_ARCH__
    return top_data.y * EXP_CUDA(extra.x * (data + offset - top_data.x) + extra.y);
  //return __fdividef(exp(data + offset - max), Dtype(K));
#else
    return top_data.y * std::exp(extra.x * (data + offset - top_data.x) + extra.y);
#endif
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_bottom_infinite(Dtype offset, typename vec<Dtype>::vec2 top_data, Dtype data, uint8_t nothing) {
    return top_data.y * ((data + offset) == top_data.x);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_offsets_finite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
                                  typename vec<Dtype>::vec2 extra) {
    return mex_backward_bottom_finite(offset, top_data, data, extra);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_offsets_infinite(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset, uint8_t nothing) {
    return mex_backward_bottom_infinite(offset, top_data, data, nothing);
}

template<typename Dtype> __device__ __host__ __forceinline__
Dtype mex_backward_epsilon(typename vec<Dtype>::vec2 top_data, Dtype data, Dtype offset,
                           Dtype epsilon) {
    const Dtype x = data + offset;
#ifdef __CUDA_ARCH__
    return top_data.y * (x * EXP_CUDA(epsilon * (x - top_data.x)) - top_data.x);
#else
    return top_data.y * (x * std::exp(epsilon * (x - top_data.x)) - top_data.x);
#endif
}

template <typename Dtype, bool REVERSE>
inline
void split_patches_cpu(const int N, const int Dim,
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
    for (int w_g = 0; w_g < W_Gs; ++w_g) {
        for (int h_g = 0; h_g < H_Gs; ++h_g) {
            for (int c_g = 0; c_g < C_Gs; ++c_g) {
                Dtype* o = out + ((c_g * H_Gs + h_g) * W_Gs + w_g) * step_out * Dim;
                const int group_addr = (c_g * group_step_c * H + h_g * group_step_h) * W + w_g * group_step_w;
                for (int l = 0; l < C_Step; ++l) {
                    for (int j = 0; j < H_Step; ++j) {
                        for (int i = 0; i < W_Step; ++i) {
                            const int base_addr_out = (l * H_Step + j) * W_Step + i;
                            const int base_addr_in  = group_addr + (l * region_step_c * H + j * region_step_h) * W  + i * region_step_w;
                            if (w_g * group_step_w + i * region_step_w >= W ||
                                h_g * group_step_h + j * region_step_h >= H ||
                                c_g * group_step_c + l * region_step_c >= C) {
                                continue;
                            }
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
            }
        }
    }
}

#endif //SIMNETS_TF_MEX_UTILS_HPP
