#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "similarity_kernel_common.hpp"
#include "im2col.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

template<typename T>
class SimilarityKernelCPU : public SimilarityKernelCommon<T> {
public:

    using Base = SimilarityKernelCommon<T>;
    using Dtype = T;

    SimilarityKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        auto input = context->input(0);
        auto templates = context->input(1);
        auto weights = context->input(2);
        const T block_out_of_bounds_value{0};

        auto input_t = input.tensor<T, 4>();
        auto templates_t = templates.tensor<T, 4>();
        auto weights_t = weights.tensor<T, 4>();

        long num_instances = templates_t.dimension(1);

        // CM NCHW <--> RM WHCN
        const int BATCH_DIM = 3;
        const int H_DIM = 1;
        const int W_DIM = 0;
        const int C_DIM = 2;
        long batch = input_t.dimension(BATCH_DIM);
        long height = input_t.dimension(H_DIM);
        long width = input_t.dimension(W_DIM);
        long channels = input_t.dimension(C_DIM);


        int block_h = this->ksize_[1];
        int block_w = this->ksize_[2];
        int block_c = this->ksize_[3];

        int stride_h = this->stride_[1];
        int stride_w = this->stride_[2];
        int stride_c = this->stride_[3];

        int64 out_h;
        int64 out_w;
        int64 out_c;

        int64 pad_h;
        int64 pad_w;
        int64 pad_c;
        GetWindowedOutputSize(height, block_h, stride_h, this->padding_, &out_h, &pad_h);
        GetWindowedOutputSize(width, block_w, stride_w, this->padding_, &out_w, &pad_w);
        GetWindowedOutputSize(channels, block_c, stride_c, this->padding_, &out_c, &pad_c);

        bool is_1x1 = block_c == channels && block_w == 1 && block_h == 1
                  && stride_h == 1 && stride_w == 1
                  && pad_c == 0 && pad_h == 0 && pad_w == 0;


        Tensor *output = NULL;

        TensorShape output_shape{out_w, out_h, out_c, batch};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output));
        auto output_t = output->tensor<T, 4>();

        long M = num_instances;
        long K = block_c * block_h * block_w;
        long N = out_h * out_w * out_c;

        Tensor col_buffer;
        TensorShape col_buffer_shape{1,
                                     block_c * block_h * block_w, out_c * out_h, out_w};

        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buffer_shape, &col_buffer));
        auto col_buffer_t = col_buffer.tensor<T, 3>();

        // TODO: Lookup SetPadding


        using Dtype = T;
        Dtype *col_buff = NULL;
        if (!is_1x1) {
            col_buff = col_buffer_t.data();
        }

        // TODO: Check that all dimensions are consistent with ggemm
        const Dtype *templates_buff = templates_t.data();
        const Dtype *weights_buff = weights_t.data();

        const int params_size = num_instances * block_w * block_h * block_c;
        const int padding_size = ggemm_padded_output_size(num_instances, block_c * block_h * block_w);

        Tensor interlaced;
        TensorShape interlaced_shape{2 * (params_size + padding_size)};
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, interlaced_shape, &interlaced));
        auto interlaced_t = col_buffer.tensor<T, 3>();

        typename vec<Dtype>::vec2 *inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_t.data());
        interlace_cpu<Dtype>(params_size, templates_buff, weights_buff, inter_params);

        const Dtype *bottom_data = input_t.data();
        Dtype *top_data = input_t.data();
        for (int n = 0; n < batch; ++n) {
            // im2col transformation: unroll input regions for filtering
            // into column matrix for multplication.
            if (!is_1x1) {
                simnets_tf::im2col_3d_cpu(
                        bottom_data + n * (height * width * channels),
                        channels, height, width,
                        block_c, block_h, block_w,
                        pad_c, pad_h, pad_w,
                        stride_c, stride_h, stride_w,
                        col_buff, true, block_out_of_bounds_value);
            } else {  // special case for 1x1 convolution
                    col_buff = input_t.data() + n * (height * width * channels);
            }

            switch (this->layer_param_.similarity_param().similarity_function()) {
                case SimilarityParameter_SimilarityFunction_CONVOLUTION:
                    ggemm_cpu
                            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                    sim_linear_forward<Dtype>, ggemm_add<Dtype>, false>
                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
                    break;
                case SimilarityParameter_SimilarityFunction_L1:
                    ggemm_cpu
                            <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                    sim_l1_forward<Dtype>, ggemm_add<Dtype>, false>
                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
                    break;
                case SimilarityParameter_SimilarityFunction_L2:
                    if (normalization_term_) {
                        if (use_log_space_weight_param_) {
                            if (ignore_nan_input_) {
                                ggemm_cpu
                                        <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                sim_l2_normalized_forward<Dtype, true, true>, ggemm_add<Dtype>, false>
                                        (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0,
                                         normalization_term_fudge_);
                            } else {
                                ggemm_cpu
                                        <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                sim_l2_normalized_forward<Dtype, true, false>, ggemm_add<Dtype>, false>
                                        (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0,
                                         normalization_term_fudge_);
                                caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                                                        top_data + top[bottom_idx]->offset(n));
                            }
                        } else {
                            if (ignore_nan_input_) {
                                ggemm_cpu
                                        <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                sim_l2_normalized_forward<Dtype, false, true>, ggemm_add<Dtype>, false>
                                        (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0,
                                         normalization_term_fudge_);
                            } else {
                                ggemm_cpu
                                        <typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                                sim_l2_normalized_forward<Dtype, false, false>, ggemm_add<Dtype>, false>
                                        (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0,
                                         normalization_term_fudge_);
                                caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                                                        top_data + top[bottom_idx]->offset(n));
                            }
                        }
                    } else {
                        if (use_log_space_weight_param_) {
                            ggemm_cpu
                                    <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                            sim_l2_forward<Dtype, true>, ggemm_add<Dtype>, false>
                                    (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
                        } else {
                            ggemm_cpu
                                    <typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                            sim_l2_forward<Dtype, false>, ggemm_add<Dtype>, false>
                                    (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(n), 0, 0);
                        }

                    }
                    break;
                default:
                    break;
            }
            // Add bias.
            if (bias_term_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_instances_,
                                      N_, 1, (Dtype) 1., this->blobs_[2]->cpu_data(),
                                      bias_multiplier_.cpu_data(),
                                      (Dtype) 1., top_data + top[bottom_idx]->offset(n));
            }
        }
    }
};






