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
        auto col_buffer_t = col_buffer.tensor<T, 4>();
        auto row_buffer_t = row_buffer.tensor<T, 4>();

        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &row_buffer));

        TensorShape interlaced_grad_shape{weights_grad_t.size() + templates_grad_t.size()};
        Tensor interlaced_grad;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, interlaced_grad_shape, &interlaced_grad));
        auto interlaced_grad_t = interlaced_grad.tensor<T, 4>();

        using Dtype = T;

        const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
        const int padding_size = ggemm_padded_output_size(num_instances_, block_c_ * block_h_ * block_w_);

        Tensor interlaced;
        TensorShape interlaced_shape{2 * (params_size + padding_size)};
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, interlaced_shape, &interlaced));
        auto interlaced_t = interlaced.tensor<T, 1>();

        typename vec<Dtype>::vec2 * inter_params = NULL;
        inter_params = reinterpret_cast<typename vec<Dtype>::vec2 *>(interlaced_t.data());
        interlace_cpu<Dtype>(M_ * K_,   templates_t.data(), weights_t.data(), inter_params);

        typename vec<Dtype>::vec2 * interlaced_params_diff = NULL;

        interlaced_params_diff = reinterpret_cast<typename vec<Dtype>::vec2 *>(interlaced_grad_t.data());
        interlace_cpu<Dtype>(params_size,
                             templates_grad_t.data(), weights_grad_t.data(),
                             interlaced_params_diff);

        const Dtype* top_diff = NULL;
        const Dtype* col_buff = NULL;
        const Dtype* bottom_data = input_t.data();
        for (int n = 0; n < batch_; ++n) {
            // Since we saved memory in the forward pass by not storing all col
            // data, we will need to recompute them.
            if (!is_1x1_) {
                simnets_tf::im2col_3d_cpu<T>(
                        bottom_data + n * (height_ * width_ * channels_),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buffer_t.data(), false, out_of_bounds_value_);
                col_buff = col_buffer_t.data();
            } else {
                    col_buff = bottom_data + n * (height_ * width_ * channels_);
            }
            Dtype* row_buff = row_buffer_t.data();
            typename TTypes<T,2>::ConstTensor colMap(col_buff, K_, N_);
            typename TTypes<T,2>::Tensor rowMap(row_buff, N_, K_);
            Eigen::array<int, 2> transposeIdx({1, 0});
            rowMap.device(context->eigen_cpu_device()) = colMap.shuffle(transposeIdx);
            top_diff = output_grad_t.data() + n * (out_c_ * out_h_ * out_w_);
            // gradient w.r.t. weights and templates. Note that we will accumulate diffs.
            switch (similarity_function_) {
                case SIM_FUNC_L1:
                    ggemm_readc_cpu
                            <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                    sim_l1_backward_weights<Dtype>, add_vec2<Dtype>, true,
                                    no_op<typename vec<Dtype>::vec2>, false>
                            (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                    break;
                case SIM_FUNC_L2:
                    if (normalization_term_) {
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
                    } else {
                        ggemm_readc_cpu
                                <false, false, Dtype, Dtype, typename vec<Dtype>::vec2, uint8_t,
                                        sim_l2_backward_weights<Dtype, false>, add_vec2<Dtype>, true,
                                        no_op<typename vec<Dtype>::vec2>, false>
                                (M_, K_, N_, top_diff, row_buff, inter_params, interlaced_params_diff, make_vec2<Dtype>(0,0), 0);
                    }
                    break;
                default:
                    break;
            }
        }
        deinterlace_cpu<Dtype>(params_size,
                               interlaced_params_diff, templates_grad_t.data(), weights_grad_t.data());

    }
};

REGISTER_KERNEL_BUILDER(
        Name("SimilarityParametersGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<double>("T"),
        SimilarityParametersGradKernelCPU<double>);
REGISTER_KERNEL_BUILDER(
        Name("SimilarityParametersGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        SimilarityParametersGradKernelCPU<float>);