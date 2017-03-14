//
// Created by elhanani on 14/03/17.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "similarity_kernel_common.hpp"
#include "im2col.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

template<typename T>
class SimilarityParametersGradKernelCPU : public SimilarityKernelCommon {
public:

    using Base = SimilarityKernelCommon;
    using Dtype = T;

    SimilarityParametersGradKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        this->CalculateDimensions<T>(context);
        auto input = context->input(0);
        auto templates = context->input(1);
        auto weights = context->input(2);
        auto output_grad = context->input(3);

        auto input_t = input.tensor<T, 4>();
        auto templates_t = templates.tensor<T, 4>();
        auto weights_t = weights.tensor<T, 4>();
        auto output_grad_t = output_grad.tensor<T, 4>();

        Tensor *weights_grad = NULL;
        Tensor *templates_grad = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, weights.shape(),
                                                         &weights_grad));
        OP_REQUIRES_OK(context, context->allocate_output(1, templates.shape(),
                                                         &templates_grad));

        auto weights_grad_t = weights_grad->tensor<T, 4>();
        auto templates_grad_t = templates_grad->tensor<T, 4>();

        Tensor col_buffer;
        Tensor row_buffer;
        int col_buffer_padding = ggemm_padded_output_size(block_c_ * block_h_ * block_w_,
                                                          out_h_ * out_w_);
        TensorShape col_buffer_shape{
                block_c_ * block_h_ * block_w_ * out_h_ * out_w_ + col_buffer_padding};

        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &row_buffer));
        using Dtype = T;

        const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
        const int padding_size = ggemm_padded_output_size(num_instances_, block_c_ * block_h_ * block_w_);

        Tensor interlaced;
        TensorShape interlaced_shape{2 * (params_size + padding_size)};
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, interlaced_shape, &interlaced));
        auto interlaced_t = interlaced.tensor<T, 1>();

        typename vec<Dtype>::vec2 * inter_params = NULL;
        inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_t.data());
        interlace_cpu<Dtype>(M_ * K_,   templates_t.data(), weights_t.data(), inter_params);


        // TODO: Continue from here
        typename vec<Dtype>::vec2 * interlaced_params_diff = NULL;

        interlaced_params_diff = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_diff_->mutable_cpu_data());
        interlace_cpu<Dtype>(params_size,
                             templates_grad_t.data(), weights_grad_t.data(),
                             interlaced_params_diff);

        const Dtype* top_diff = NULL;
        Dtype* col_buff = NULL;
        if (!is_1x1_ || normalize_patches_) {
            col_buff = col_buffer_.mutable_cpu_data();
        }
        const Dtype* bottom_data = bottom[top_idx]->cpu_data();
        Dtype* bottom_diff = bottom[top_idx]->mutable_cpu_diff();
        for (int n = 0; n < num_; ++n) {
            // Since we saved memory in the forward pass by not storing all col
            // data, we will need to recompute them.
            if (!is_1x1_) {
                im2col_3d_cpu(
                        bottom_data + bottom[top_idx]->offset(n),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buff, true, block_out_of_bounds_value_);
            } else {
                if (!normalize_patches_) {
                    col_buff = bottom[top_idx]->mutable_cpu_data() + bottom[top_idx]->offset(n);
                } else {
                    caffe_copy(N_ * K_, bottom[top_idx]->mutable_cpu_data() + bottom[top_idx]->offset(n), col_buff);
                }
            }
            Dtype* row_buff = row_buffer_.mutable_cpu_data();
            caffe_cpu_transpose(K_, N_,
                                col_buff,
                                row_buff);
            if (normalize_patches_) {
                caffe_copy(K_ * N_,
                           row_buff,
                           row_buffer_.mutable_cpu_diff());
                caffe_cpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
                                                         row_buff, normalize_variance_);
                caffe_cpu_transpose(N_, K_,
                                    row_buff,
                                    col_buff);
            }
            top_diff = top[top_idx]->cpu_diff() + top[0]->offset(n);
            // gradient w.r.t. weights and templates. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0] || this->param_propagate_down_[1]) {
                switch (this->layer_param_.similarity_param().similarity_function()) {
                    case SimilarityParameter_SimilarityFunction_CONVOLUTION:
                        ggemm_readc_cpu
                                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                        sim_linear_backward_weights<Dtype>, add_vec2<Dtype>, true,
                                        no_op<typename vec<Dtype>::vec2>, false>
                                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L1:
                        ggemm_readc_cpu
                                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                        sim_l1_backward_weights<Dtype>, add_vec2<Dtype>, true,
                                        no_op<typename vec<Dtype>::vec2>, false>
                                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L2:
                        if (normalization_term_) {
                            if (use_log_space_weight_param_) {
                                if (ignore_nan_input_) {
                                    ggemm_readc_cpu
                                            <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                                                    sim_l2_normalized_backward_weights<Dtype, true, true>, add_vec2<Dtype>, true,
                                                    no_op<typename vec<Dtype>::vec2>, false>
                                            (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                                             normalization_term_fudge_);
                                } else {
                                    ggemm_readc_cpu
                                            <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                                                    sim_l2_normalized_backward_weights<Dtype, true, false>, add_vec2<Dtype>, true,
                                                    no_op<typename vec<Dtype>::vec2>, false>
                                            (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                                             normalization_term_fudge_);
                                }
                            } else {
                                if (ignore_nan_input_) {
                                    ggemm_readc_cpu
                                            <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                                                    sim_l2_normalized_backward_weights<Dtype, false, true>, add_vec2<Dtype>, true,
                                                    no_op<typename vec<Dtype>::vec2>, false>
                                            (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                                             normalization_term_fudge_);
                                } else {
                                    ggemm_readc_cpu
                                            <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, Dtype,
                                                    sim_l2_normalized_backward_weights<Dtype, false, false>, add_vec2<Dtype>, true,
                                                    no_op<typename vec<Dtype>::vec2>, false>
                                            (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0),
                                             normalization_term_fudge_);
                                }
                            }
                        } else {
                            if (use_log_space_weight_param_) {
                                ggemm_readc_cpu
                                        <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                                sim_l2_backward_weights<Dtype, true>, add_vec2<Dtype>, true,
                                                no_op<typename vec<Dtype>::vec2>, false>
                                        (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                            } else {
                                ggemm_readc_cpu
                                        <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                                sim_l2_backward_weights<Dtype, false>, add_vec2<Dtype>, true,
                                                no_op<typename vec<Dtype>::vec2>, false>
                                        (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                            }
                        }
                        break;
                    default:
                        break;
                }
            }

            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[top_idx]) {
                Dtype* col_diff_buff = NULL;
                if (is_1x1_) {
                    col_diff_buff = bottom[top_idx]->mutable_cpu_diff() + bottom[top_idx]->offset(n);
                } else {
                    col_diff_buff = col_buffer_.mutable_cpu_diff();
                }

                switch (this->layer_param_.similarity_param().similarity_function()) {
                    case SimilarityParameter_SimilarityFunction_CONVOLUTION:
                        ggemm_readc_cpu
                                <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                        sim_linear_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L1:
                        ggemm_readc_cpu
                                <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                        sim_l1_backward_bottom<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L2:
                        if (normalization_term_) {
                            if (use_log_space_weight_param_) {
                                if (ignore_nan_input_) {
                                    ggemm_readc_cpu
                                            <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                    sim_l2_normalized_backward_bottom<Dtype, true, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                            (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                                } else {
                                    ggemm_readc_cpu
                                            <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                    sim_l2_normalized_backward_bottom<Dtype, true, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                            (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                                }
                            } else {
                                if (ignore_nan_input_) {
                                    ggemm_readc_cpu
                                            <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                    sim_l2_normalized_backward_bottom<Dtype, false, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                            (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                                } else {
                                    ggemm_readc_cpu
                                            <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                    sim_l2_normalized_backward_bottom<Dtype, false, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                            (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, normalization_term_fudge_);
                                }
                            }
                        } else {
                            if (use_log_space_weight_param_) {
                                ggemm_readc_cpu
                                        <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                                sim_l2_backward_bottom<Dtype, true>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                        (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                            } else {
                                ggemm_readc_cpu
                                        <true, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                                sim_l2_backward_bottom<Dtype, false>, ggemm_add<Dtype>, false, no_op<Dtype>, false>
                                        (K_, N_, M_, inter_params, top_diff, col_buff, col_diff_buff, 0, 0);
                            }
                        }
                        break;
                    default:
                        break;
                }

                // col2im back to the data
                if (!is_1x1_) {
                    col2im_3d_cpu(
                            col_diff_buff,
                            channels_, height_, width_,
                            block_c_, block_h_, block_w_,
                            pad_c_, pad_h_, pad_w_,
                            stride_c_, stride_h_, stride_w_,
                            bottom_diff + bottom[top_idx]->offset(n));
                }
            }
        }
        const int params_size = M_ * K_;
        deinterlace_cpu<Dtype>(params_size,
                               interlaced_params_diff, templates_diff, weights_diff);

    }
};