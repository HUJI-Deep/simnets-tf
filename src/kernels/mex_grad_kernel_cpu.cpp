//
// Created by elhanani on 01/04/17.
//

#include "kernels/mex_kernel_common.hpp"
#include "utils/ggemm_cpu.hpp"

using namespace tensorflow;

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
class MEXGradInputKernelCPU : public MEXKernelCommon {
public:

    using Base = MEXKernelCommon;
    using Dtype = T;

    MEXGradInputKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        CalculateDimensionsWithConext(context);

        auto input = context->input(0);
        auto offsets_unpadded = context->input(1);
        auto input_t = input.tensor<T, 4>();
        auto output = context->input(2);
        auto output_t = output.tensor<T, 4>();

        auto output_grad = context->input(3);
        auto output_grad_t = output_grad.tensor<T, 4>();
        auto offsets_unpadded_t = offsets_unpadded.tensor<T, 5>();

        Tensor offsets_padded, offsets_padded_transpose;
        TensorShape offsets_padded_shape{{offsets_unpadded_t.size() + ggemm_padded_output_size(M_, K_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, offsets_padded_shape, &offsets_padded);
        context->allocate_temp(DataTypeToEnum<T>::value, offsets_padded_shape, &offsets_padded_transpose);
        auto offsets_padded_t = offsets_padded.tensor<T, 1>();
        auto offsets_padded_transpose_t = offsets_padded_transpose.tensor<T, 1>();
        copy_with_eigen(offsets_padded_t.data(), offsets_unpadded_t.data(),
                        offsets_unpadded_t.size(), context->eigen_cpu_device());


        Tensor *input_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(), &input_grad));
        auto input_grad_t = input_grad->tensor<T, 4>();

        Tensor col_buffer, col_buffer_grad;
        TensorShape col_buffer_shape{{K_ * channels_out_ * height_out_ * width_out_ + ggemm_padded_output_size(K_, N_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer);
        context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer_grad);
        auto col_buffer_t = col_buffer.tensor<T, 1>();
        auto col_buffer_grad_t = col_buffer_grad.tensor<T, 1>();

        Tensor split_patches_in_tensor, split_patches_in_grad_tensor;
        TensorShape split_patches_in_shape{{num_regions_ * K_ * region_size_ + ggemm_padded_output_size(K_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_in_shape, &split_patches_in_tensor);
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_in_shape, &split_patches_in_grad_tensor);
        auto split_patches_in_t = split_patches_in_tensor.tensor<T, 1>();
        auto split_patches_in_grad_t = split_patches_in_grad_tensor.tensor<T, 1>();

        Tensor split_patches_out_tensor, split_patches_out_grad_tensor;
        TensorShape split_patches_out_shape{{num_regions_ * M_ * region_size_ + ggemm_padded_output_size(M_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_out_shape, &split_patches_out_tensor);
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_out_shape, &split_patches_out_grad_tensor);
        auto split_patches_out_t = split_patches_out_tensor.tensor<T, 1>();
        auto split_patches_out_grad_t = split_patches_out_grad_tensor.tensor<T, 1>();

        Tensor split_patches_inter_tensor;
        TensorShape split_patches_inter_tensor_shape{{2*(num_regions_ * M_ * region_size_ + ggemm_padded_output_size(M_, region_size_))}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_inter_tensor_shape, &split_patches_inter_tensor);
        auto split_patches_inter_t = split_patches_inter_tensor.tensor<T, 1>();

        // -------------------------------------------------------------------------------

        const Dtype* transposed_offsets = NULL;
        transposed_offsets = static_cast<const Dtype*>(offsets_padded_transpose_t.data());
        {
            typename TTypes<T, 3>::ConstTensor tmp_offsets(offsets_padded_t.data(), num_regions_, M_, K_);
            typename TTypes<T, 3>::Tensor tmp_offsets_transpose(offsets_padded_transpose_t.data(), num_regions_, K_, M_);
            Eigen::array<int, 3> indices{0,2,1};
            tmp_offsets_transpose.device(context->eigen_cpu_device()) = tmp_offsets.shuffle(indices);
        }

        const Dtype* top_diff = output_grad_t.data();
        const Dtype* top_data = output_t.data();
        Dtype* col_buff = NULL;
        Dtype* col_diff = NULL;
        if (!is_1x1_) {
            col_buff = col_buffer_t.data();
        }
        if (!is_1x1_) {
            col_diff = col_buffer_grad_t.data();
        }

        auto input_at_batch = [&](int n) {
            return input_t.data() + n * channels_ * height_ * width_;
        };

        auto input_grad_at_batch = [&](int n) {
            return input_grad_t.data() + n * channels_ * height_ * width_;
        };

        for (int n = 0; n < batch_; ++n) {
            // Since we saved memory in the forward pass by not storing all col
            // data, we will need to recompute them.
            if (!is_1x1_) {
                simnets_tf::im2col_3d_cpu<T>(
                        input_at_batch(n),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buff,
                        blocks_round_down_, blocks_out_of_bounds_value_);
            } else {  // special case for 1x1 convolution
                col_diff = input_grad_at_batch(n);
                col_buff =  input_at_batch(n);
            }

            // Prepare input for backprop
            const Dtype* current_top_data = top_data + n * M_ * N_;
            const Dtype* current_top_diff = top_diff + n * M_ * N_;
            Dtype *split_patches_in, *split_patches_in_diff;
            Dtype *split_patches_out, *split_patches_out_diff;
            typename vec<T>::vec2 *split_patches_out_inter;

            if (num_regions_ > 1) {
                split_patches_in = split_patches_in_t.data();
                split_patches_in_diff = split_patches_in_grad_t.data();
                split_patches_out = split_patches_out_t.data();
                split_patches_out_diff = split_patches_out_grad_t.data();
                split_patches_cpu<Dtype, false>(N_, K_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                col_buff, split_patches_in, use_unshared_regions_);
                split_patches_cpu<Dtype, false>(N_, M_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                current_top_data, split_patches_out, use_unshared_regions_);
                split_patches_cpu<Dtype, false>(N_, M_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                current_top_diff, split_patches_out_diff, use_unshared_regions_);
            } else {
                split_patches_in = col_buff;
                split_patches_in_diff = col_diff;
                split_patches_out = (Dtype*)current_top_data;
                split_patches_out_diff = (Dtype*)current_top_diff;
            }
            split_patches_out_inter = reinterpret_cast<typename vec<Dtype>::vec2 *>(
                    split_patches_inter_t.data());
            interlace_cpu(num_regions_ * M_ * region_size_, split_patches_out, split_patches_out_diff,
                          split_patches_out_inter);
            // Caculate backprop
            if (std::isfinite(epsilon_)) {
                ggemm_readc_cpu
                        <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, typename vec<Dtype>::vec2,
                                mex_backward_bottom_finite<Dtype>, ggemm_add<Dtype>, false,
                                no_op<Dtype, typename vec<Dtype>::vec2>, false,
                                true, true, true>
                        (K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
                         split_patches_in, split_patches_in_diff, 0,
                         make_vec2<Dtype>(epsilon_, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
            } else {
                ggemm_readc_cpu
                        <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, uint8_t,
                                mex_backward_bottom_infinite<Dtype>, ggemm_add<Dtype>, false,
                                no_op<Dtype, uint8_t>, false,
                                true, true, true>
                        (K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
                         split_patches_in, split_patches_in_diff, 0, 0, num_regions_);
            }
            // Copy to bottom if needed
            if (num_regions_ > 1) {
                split_patches_cpu<Dtype, true>(N_, K_,
                                               width_out_, height_out_, channels_out_,
                                               offsets_w_, offsets_h_, offsets_c_,
                                               shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                               col_diff, split_patches_in_diff, use_unshared_regions_);
            }

            if (!is_1x1_) {
                simnets_tf::col2im_3d_cpu<T>(
                        col_diff,
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        input_grad_at_batch(n),
                        blocks_round_down_);
            }
        }
    }
};

template<typename T>
class MEXGradOffsetsKernelCPU : public MEXKernelCommon {
public:

    using Base = MEXKernelCommon;
    using Dtype = T;

    MEXGradOffsetsKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        CalculateDimensionsWithConext(context);

        auto input = context->input(0);
        auto offsets_unpadded = context->input(1);
        auto input_t = input.tensor<T, 4>();
        auto output = context->input(2);
        auto output_t = output.tensor<T, 4>();

        auto output_grad = context->input(3);
        auto output_grad_t = output_grad.tensor<T, 4>();
        auto offsets_unpadded_t = offsets_unpadded.tensor<T, 5>();

        auto zero_out = [&](Tensor& t)
        {
            auto flat = t.flat<T>();
            flat.device(context->eigen_cpu_device()) = flat.constant(0);
        };

        Tensor offsets_padded;
        TensorShape offsets_padded_shape{{offsets_unpadded_t.size() + ggemm_padded_output_size(M_, K_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, offsets_padded_shape, &offsets_padded);
        auto offsets_padded_t = offsets_padded.tensor<T, 1>();
        copy_with_eigen(offsets_padded_t.data(), offsets_unpadded_t.data(),
                        offsets_unpadded_t.size(), context->eigen_cpu_device());


        Tensor *offsets_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, context->input(1).shape(), &offsets_grad));
        auto offsets_grad_t = offsets_grad->tensor<T, 5>();
        //offsets_grad_t.device(context->eigen_cpu_device()) = offsets_grad_t.constant(0);
        zero_out(*offsets_grad);

        Tensor col_buffer, col_buffer_grad;
        TensorShape col_buffer_shape{{K_ * channels_out_ * height_out_ * width_out_ + ggemm_padded_output_size(K_, N_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer);
        context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer_grad);
        auto col_buffer_t = col_buffer.tensor<T, 1>();
        zero_out(col_buffer);

        Tensor split_patches_in_tensor, split_patches_in_grad_tensor;
        TensorShape split_patches_in_shape{{num_regions_ * K_ * region_size_ + ggemm_padded_output_size(K_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_in_shape, &split_patches_in_tensor);
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_in_shape, &split_patches_in_grad_tensor);
        zero_out(split_patches_in_tensor);
        zero_out(split_patches_in_grad_tensor);
        auto split_patches_in_t = split_patches_in_tensor.tensor<T, 1>();
        auto split_patches_in_grad_t = split_patches_in_grad_tensor.tensor<T, 1>();

        Tensor split_patches_out_tensor, split_patches_out_grad_tensor;
        TensorShape split_patches_out_shape{{num_regions_ * M_ * region_size_ + ggemm_padded_output_size(M_, region_size_)}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_out_shape, &split_patches_out_tensor);
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_out_shape, &split_patches_out_grad_tensor);
        zero_out(split_patches_out_tensor);
        zero_out(split_patches_out_grad_tensor);
        auto split_patches_out_t = split_patches_out_tensor.tensor<T, 1>();
        auto split_patches_out_grad_t = split_patches_out_grad_tensor.tensor<T, 1>();

        Tensor split_patches_inter_tensor;
        TensorShape split_patches_inter_tensor_shape{{2*(num_regions_ * M_ * region_size_ + ggemm_padded_output_size(M_, region_size_))}};
        context->allocate_temp(DataTypeToEnum<T>::value, split_patches_inter_tensor_shape, &split_patches_inter_tensor);
        auto split_patches_inter_t = split_patches_inter_tensor.tensor<T, 1>();

        // -------------------------------------------------------------------------------

        const Dtype* top_diff = output_grad_t.data();
        const Dtype* top_data = output_t.data();
        Dtype* col_buff = NULL;
        if (!is_1x1_) {
            col_buff = col_buffer_t.data();
        }

        auto input_at_batch = [&](int n) {
            return input_t.data() + n * channels_ * height_ * width_;
        };

        for (int n = 0; n < batch_; ++n) {
            // Since we saved memory in the forward pass by not storing all col
            // data, we will need to recompute them.
            if (!is_1x1_) {
                simnets_tf::im2col_3d_cpu<T>(
                        input_at_batch(n),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buff,
                        blocks_round_down_, blocks_out_of_bounds_value_);
            } else {  // special case for 1x1 convolution
                col_buff =  input_at_batch(n);
            }

            // Prepare input for backprop
            const Dtype* current_top_data = top_data + n * M_ * N_;
            const Dtype* current_top_diff = top_diff + n * M_ * N_;
            Dtype *split_patches_in;
            Dtype *split_patches_out, *split_patches_out_diff;
            typename vec<T>::vec2 *split_patches_out_inter;

            if (num_regions_ > 1) {
                split_patches_in = split_patches_in_t.data();
                split_patches_out = split_patches_out_t.data();
                split_patches_out_diff = split_patches_out_grad_t.data();
                split_patches_cpu<Dtype, false>(N_, K_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                col_buff, split_patches_in, use_unshared_regions_);
                split_patches_cpu<Dtype, false>(N_, M_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                current_top_data, split_patches_out, use_unshared_regions_);
                split_patches_cpu<Dtype, false>(N_, M_,
                                                width_out_, height_out_, channels_out_,
                                                offsets_w_, offsets_h_, offsets_c_,
                                                shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
                                                current_top_diff, split_patches_out_diff, use_unshared_regions_);
            } else {
                split_patches_in = col_buff;
                split_patches_out = (Dtype*)current_top_data;
                split_patches_out_diff = (Dtype*)current_top_diff;
            }
            split_patches_out_inter = reinterpret_cast<typename vec<Dtype>::vec2 *>(
                    split_patches_inter_t.data());
            interlace_cpu(num_regions_ * M_ * region_size_, split_patches_out, split_patches_out_diff,
                          split_patches_out_inter);

            // temp use of split_patches_in_diff for transposing the patches
            {
                typename TTypes<T, 3>::ConstTensor tmp_patches(split_patches_in, num_regions_, K_, region_size_);
                typename TTypes<T, 3>::Tensor tmp_patches_transpose(split_patches_in_grad_t.data(), num_regions_, region_size_, K_);
                Eigen::array<int, 3> indices{0,2,1};
                tmp_patches_transpose.device(context->eigen_cpu_device()) = tmp_patches.shuffle(indices);
            }
            if (std::isfinite(epsilon_)) {
                ggemm_readc_cpu
                        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, typename vec<Dtype>::vec2,
                                mex_backward_offsets_finite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, typename vec<Dtype>::vec2>, false,
                                true, true, true>
                        (M_, K_, region_size_, split_patches_out_inter, split_patches_in_grad_t.data(),
                         offsets_padded_t.data(), offsets_grad_t.data(), 0,
                         make_vec2<Dtype>(epsilon_, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
            } else {
                ggemm_readc_cpu
                        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                mex_backward_offsets_infinite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, uint8_t>, false,
                                true, true, true>
                        (M_, K_, region_size_, split_patches_out_inter, split_patches_in_grad_t.data(),
                         offsets_padded_t.data(), offsets_grad_t.data(), 0, 0, num_regions_);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("MexInputGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        MEXGradInputKernelCPU<float>);
REGISTER_KERNEL_BUILDER(
        Name("MexInputGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<double>("T"),
        MEXGradInputKernelCPU<double>);
REGISTER_KERNEL_BUILDER(
        Name("MexOffsetsGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        MEXGradOffsetsKernelCPU<float>);
REGISTER_KERNEL_BUILDER(
        Name("MexOffsetsGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<double>("T"),
        MEXGradOffsetsKernelCPU<double>);