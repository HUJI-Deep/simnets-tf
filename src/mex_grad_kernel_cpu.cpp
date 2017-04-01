//
// Created by elhanani on 01/04/17.
//

#include "mex_kernel_common.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

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
        auto input_grad_t = output->input_grad<T, 4>();

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

        const Dtype* offsets = offsets_padded_t.data();
        const Dtype* transposed_offsets = NULL;
        transposed_offsets = static_cast<const Dtype*>(offsets_padded_transpose_t.data());

        // TODO: Transpose with eigen shuffle
        for (int r = 0; r < num_regions_; ++r) {
            const int offsets_idx = r * M_ * K_;
            caffe_cpu_transpose(M_, K_,
                                offsets + offsets_idx,
                                static_cast<Dtype*>(transposed_offsets_->mutable_cpu_data()) + offsets_idx);
        }

        const Dtype* top_diff = input_grad_t.data();
        const Dtype* top_data = input_t.data();
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

        auto output_at_batch = [&](int n) {
            return output_t.data() + n * channels_out_total_ * height_out_ * width_out_;
        };

        const Dtype* bottom_data = input_t.data();
        Dtype* bottom_diff = output_t.data();
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
                col_diff = bottom_diff + bottom[bottom_idx]->offset(n);
                col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
            }

            // Prepare input for backprop
            const Dtype* current_top_data = top_data + n * M_ * N_;
            const Dtype* current_top_diff = top_diff + n * M_ * N_;
            if (num_regions_ > 1) {
                split_patches_in = split_patches_in_.mutable_cpu_data();
                split_patches_in_diff = split_patches_in_.mutable_cpu_diff();
                split_patches_out = split_patches_out_.mutable_cpu_data();
                split_patches_out_diff = split_patches_out_.mutable_cpu_diff();
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
            split_patches_out_inter = static_cast<typename vec<Dtype>::vec2 *>(
                    split_patches_out_inter_->mutable_cpu_data());
            interlace_cpu(num_regions_ * M_ * region_size_, split_patches_out, split_patches_out_diff,
                          split_patches_out_inter);
            // Caculate backprop
            if (std::isfinite(epsilon)) {
                ggemm_readc_cpu
                        <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, typename vec<Dtype>::vec2,
                                mex_backward_bottom_finite<Dtype>, ggemm_add<Dtype>, false,
                                no_op<Dtype, typename vec<Dtype>::vec2>, false,
                                true, true, true>
                        (K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
                         split_patches_in, split_patches_in_diff, 0,
                         make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
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
                col2im_3d_cpu(
                        col_diff,
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        bottom_diff + bottom[bottom_idx]->offset(n),
                        blocks_round_down_);
            }
        }
    }
};