//
// Created by elhanani on 01/04/17.
//

#include "mex_kernel_common.hpp"
#include "ggemm_cpu.hpp"
#include "ggemm.cuh"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition;\
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace {
    // CUDA: use 512 threads per block
    const int SIMNETS_TF_CUDA_NUM_THREADS = 512;

    // CUDA: number of blocks for threads.
    inline int SIMNETS_TF_GET_BLOCKS(const int N) {
        return (N + SIMNETS_TF_CUDA_NUM_THREADS - 1) / SIMNETS_TF_CUDA_NUM_THREADS;
    }
}
using namespace tensorflow;
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
        if (w_g * W_Step + i < W &&
            h_g * H_Step + j < H &&
            c_g * C_Step + l < C) {
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
void split_patches_gpu(const int N, const int Dim,
                       const int W, const int H, const int C,
                       const int W_Gs, const int H_Gs, const int C_Gs,
                       const int W_Step, const int H_Step, const int C_Step,
                       typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
                       Dtype* out, const bool use_unshared_regions) {
    const int num_kernels = W_Step * H_Step * C_Step * W_Gs * H_Gs * C_Gs;
    // NOLINT_NEXT_LINE(whitespace/operators)
    split_patches_kernel<Dtype, REVERSE><<<SIMNETS_TF_GET_BLOCKS(num_kernels), SIMNETS_TF_CUDA_NUM_THREADS>>>(
            num_kernels, N, Dim, W, H, C, W_Gs, H_Gs, C_Gs, W_Step, H_Step, C_Step, in, out, use_unshared_regions);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void mex_forward_gpu(const int M, const int N, const int K, const bool softmax_mode,
                     const Dtype epsilon, const Dtype* offsets, const Dtype* in, Dtype* out, const int batch_size = 1) {
    const Dtype init_value = epsilon > 0 ? -INFINITY : INFINITY;
    if (epsilon > 0) {
        ggemm_gpu
                <Dtype, Dtype, Dtype, uint8_t,
                        ggemm_add<Dtype, uint8_t>, ggemm_max<Dtype>, false,
                        true, true, true>
                (M, N, K, offsets, in, out,
                 init_value, init_value, init_value, 0, batch_size);
    } else {
        ggemm_gpu
                <Dtype, Dtype, Dtype, uint8_t,
                        ggemm_add<Dtype, uint8_t>, ggemm_min<Dtype>, false,
                        true, true, true>
                (M, N, K, offsets, in, out,
                 init_value, init_value, init_value, 0, batch_size);
    }
    if (std::isfinite(epsilon)) {
        ggemm_readc_gpu
                <false, false, Dtype, Dtype, Dtype, typename vec<Dtype>::vec2,
                        mex_forward_exp<Dtype>, ggemm_add<Dtype>, true, mex_forward_out<Dtype>, true,
                        true, true, true>
                (M, N, K, offsets, in, out, out,
                 init_value, init_value, 0, make_vec2<Dtype>(epsilon, softmax_mode ? Dtype(0) : (Dtype)-std::log(K)), batch_size);
    }
}
namespace
{
    template <typename T, typename D>
    void copy_with_eigen(T* dest, const T* source, size_t sz, const D& eigen_device)
    {
        typename TTypes<T,1>::ConstTensor src(source, sz);
        typename TTypes<T,1>::Tensor dst(dest, sz);
        dst.device(eigen_device) = src;
    }
}

template<typename T>
class MEXKernelGPU : public MEXKernelCommon {
public:

    using Base = MEXKernelCommon;
    using Dtype = T;

    MEXKernelGPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        CalculateDimensionsWithConext(context);

        auto input = context->input(0);
        auto offsets_unpadded = context->input(1);
        auto input_t = input.tensor<T, 4>();
        auto offsets_unpadded_t = offsets_unpadded.tensor<T, 5>();

        Tensor offsets_padded;
        TensorShape offsets_padded_shape{{offsets_unpadded_t.size() + ggemm_padded_output_size(M_, K_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, offsets_padded_shape, &offsets_padded);
        auto offsets_padded_t = offsets_padded.tensor<T, 1>();
        copy_with_eigen(offsets_padded_t.data(), offsets_unpadded_t.data(),
                        offsets_unpadded_t.size(), context->eigen_gpu_device());


        Tensor *output = NULL;

        TensorShape output_shape{batch_, channels_out_total_, height_out_, width_out_};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto output_t = output->tensor<T, 4>();

        Tensor col_buffer;
        TensorShape col_buffer_shape{{K_ * channels_out_ * height_out_ * width_out_ + ggemm_padded_output_size(K_, N_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer);
        auto col_buffer_t = col_buffer.tensor<T, 1>();

        Tensor split_patches_in_tensor;
        TensorShape split_patches_in_shape{{num_regions_ * K_ * region_size_ + ggemm_padded_output_size(K_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_in_shape, &split_patches_in_tensor);
        auto split_patches_in_t = split_patches_in_tensor.tensor<T, 1>();

        Tensor split_patches_out_tensor;
        TensorShape split_patches_out_shape{{num_regions_ * M_ * region_size_ + ggemm_padded_output_size(M_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_out_shape, &split_patches_out_tensor);
        auto split_patches_out_t = split_patches_out_tensor.tensor<T, 1>();

        // -------------------------------------------------------------------------------

        const Dtype *col_buff = NULL;

        if (!is_1x1_) {
            col_buff = col_buffer_t.data();
        }
        const Dtype epsilon = epsilon_;
        const Dtype *split_patches_in = NULL;
        Dtype *split_patches_out = NULL;
        const Dtype *offsets = offsets_padded_t.data();

        auto input_at_batch = [&](int n) {
            return input_t.data() + n * channels_ * height_ * width_;
        };

        auto output_at_batch = [&](int n) {
            return output_t.data() + n * channels_out_total_ * height_out_ * width_out_;
        };

        for (int n = 0; n < batch_; ++n) {
            // im2col transformation: unroll input regions for filtering
            // into column matrix for multplication.
            if (!is_1x1_) {
                simnets_tf::im2col_3d_gpu<T>(
                        input_at_batch(n),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buffer_t.data(),
                        blocks_round_down_, blocks_out_of_bounds_value_);
            } else {  // special case for 1x1 convolution
                col_buff = input_at_batch(n);
            }
            // Prepare input
            Dtype* current_top = output_at_batch(n);
            if (num_regions_ > 1) {
                split_patches_in = split_patches_in_t.data();
                split_patches_out = split_patches_out_t.data();
                split_patches_gpu<Dtype, false>(N_, K_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                col_buff, split_patches_in_t.data(), use_unshared_regions_);
            } else {
                split_patches_in = col_buff;
                split_patches_out = current_top;
            }

            // Calculate
            mex_forward_gpu<Dtype>(M_, region_size_, K_, softmax_mode_, epsilon,
                                   offsets, split_patches_in, split_patches_out, num_regions_);
            // Copy to output if needed
            if (num_regions_ > 1) {
                split_patches_gpu<Dtype, true>(N_, M_,
                                               width_out_, height_out_, channels_out_,
                                               offsets_w_, offsets_h_, offsets_c_,
                                               shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                               current_top, split_patches_out, use_unshared_regions_);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("Mex")
                .Device(DEVICE_GPU)
                .TypeConstraint<float>("T"),
        MEXKernelGPU<float>);
REGISTER_KERNEL_BUILDER(
        Name("Mex")
                .Device(DEVICE_GPU)
                .TypeConstraint<double>("T"),
        MEXKernelGPU<double>);
