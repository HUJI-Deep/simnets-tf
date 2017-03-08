#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "similarity_kernel_common.hpp"
#include "im2col.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

template<typename T>
class SimilarityKernelCPU : public SimilarityKernelCommon {
public:

    using Base = SimilarityKernelCommon;
    using Dtype = T;

    SimilarityKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        this->CalculateDimensions<T>(context);
        auto input = context->input(0);
        auto templates = context->input(1);
        auto weights = context->input(2);
        const T block_out_of_bounds_value{0};

        auto input_t = input.tensor<T, 4>();
        auto templates_t = templates.tensor<T, 4>();
        auto weights_t = weights.tensor<T, 4>();

        Tensor *output = NULL;

        TensorShape output_shape{batch_, out_c_, out_h_, out_w_};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output));
        auto output_t = output->tensor<T, 4>();

        Tensor col_buffer;
        int col_buffer_padding = ggemm_padded_output_size(block_c_ * block_h_ * block_w_,
                                                          out_h_ * out_w_);
        TensorShape col_buffer_shape{
                block_c_ * block_h_ * block_w_ * out_h_ * out_w_ + col_buffer_padding};


        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
        auto col_buffer_t = col_buffer.tensor<T, 1>();

        using Dtype = T;
        Dtype *col_buff = NULL;
        if (!is_1x1_) {
            col_buff = col_buffer_t.data();
        }

        // TODO: Check that all dimensions are consistent with ggemm
        const Dtype *templates_buff = templates_t.data();
        const Dtype *weights_buff = weights_t.data();

        const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
        const int padding_size = ggemm_padded_output_size(num_instances_, block_c_ * block_h_ * block_w_);

        Tensor interlaced;
        TensorShape interlaced_shape{2 * (params_size + padding_size)};
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, interlaced_shape, &interlaced));
        auto interlaced_t = interlaced.tensor<T, 1>();

        typename vec<Dtype>::vec2 *inter_params = reinterpret_cast<typename vec<Dtype>::vec2 *>(interlaced_t.data());
        interlace_cpu<Dtype>(params_size, templates_buff, weights_buff, inter_params);

        const Dtype *bottom_data = input_t.data();
        Dtype *top_data = output_t.data();
        for (int n = 0; n < batch_; ++n) {
            // im2col transformation: unroll input regions for filtering
            // into column matrix for multplication.
            if (!is_1x1_) {
                simnets_tf::im2col_3d_cpu(
                        bottom_data + n * (height_ * width_ * channels_),
                        channels_, height_, width_,
                        block_c_, block_h_, block_w_,
                        pad_c_, pad_h_, pad_w_,
                        stride_c_, stride_h_, stride_w_,
                        col_buff, true, block_out_of_bounds_value);
            } else {  // special case for 1x1 convolution
                col_buff = input_t.data() + n * (height_ * width_ * channels_);
            }

            switch (this->similarity_function_) {
                case SIM_FUNC_L1:
                    ggemm_cpu
                            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                    sim_l1_forward<Dtype>, ggemm_add<Dtype>, false>
                            (M_, N_, K_, inter_params, col_buff, top_data + n * (out_c_ * out_h_ * out_w_), 0, 0);
                    break;
                case SIM_FUNC_L2:
                    if (this->normalization_term_) {
                        if (this->ignore_nan_input_) {
                            ggemm_cpu
                                    <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward<Dtype, false, true>, ggemm_add<Dtype>, false>
                                    (M_, N_, K_, inter_params, col_buff, top_data + n * (out_c_ * out_h_ * out_w_), 0,
                                     this->normalization_term_fudge_);
                        } else {
                            ggemm_cpu
                                    <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward<Dtype, false, false>, ggemm_add<Dtype>, false>
                                    (M_, N_, K_, inter_params, col_buff, top_data + n * (out_c_ * out_h_ * out_w_), 0,
                                     this->normalization_term_fudge_);
                            // We need to add the normalization term, we'll do it for the entire output, outside the loop
                        }
                    } else {
                        ggemm_cpu
                                <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                        sim_l2_forward<Dtype, false>, ggemm_add<Dtype>, false>
                                (M_, N_, K_, inter_params, col_buff, top_data + n * (out_c_ * out_h_ * out_w_), 0, 0);
                    }
                    break;
                default:
                    break;
            }

            if (this->similarity_function_ == SIM_FUNC_L2 && this->normalization_term_ && !this->ignore_nan_input_) {
                output_t.device(context->eigen_cpu_device()) += output_t.constant(Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI));
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
        Name("Similarity")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        SimilarityKernelCPU<float>);
REGISTER_KERNEL_BUILDER(
        Name("Similarity")
                .Device(DEVICE_CPU)
                .TypeConstraint<double>("T"),
        SimilarityKernelCPU<double>);






