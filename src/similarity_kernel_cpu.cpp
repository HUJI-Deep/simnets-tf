#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "similarity_kernel_common.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

template<typename T>
class SimilarityKernelCPU : public SimilarityKernelCommon<T> {
public:

    using Base = SimilarityKernelCommon<T>;
    SimilarityKernelCPU(OpKernelConstruction *context) : Base(context) {}

    typename TTypes<T, 4>::ConstTensor Im2Col(const typename TTypes<T, 4>::Tensor& input) {
        long rows = this->ksize_[1];
        long cols = this->ksize_[2];
        long row_stride = this->stride_[1];
        long col_stride = this->stride_[2];
        auto padding = (this->padding_ == SAME)? Eigen::PADDING_SAME : Eigen::PADDING_VALID;
        T padding_val{0};

        typename TTypes<T, 5>::Tensor res = input.extract_image_patches(rows, cols, row_stride, col_stride, /*in_row_stride*/ 1, /*in_col_stride*/ 1, padding, padding_val);
        return res;
    }


    void Compute(OpKernelContext *context) override {
        auto input = context->input(0);
        auto templates = context->input(1);
        auto weights = context->input(2);

        auto input_t = input.tensor<T, 4>();
        auto templates_t = templates.tensor<T, 4>();
        auto weights_t = weights.tensor<T, 4>();

        Tensor* output = NULL;

        int64 out_row_dim, out_col_dim;
        int64 out_row_pad, out_col_pad;
        GetWindowedOutputSize(input_t.dimension(1), ksize_[1], stride_[1], padding_, &out_row_dim, &out_row_pad);
        GetWindowedOutputSize(input_t.dimension(2), ksize_[2], stride_[2], padding_, &out_col_dim, &out_col_pad);

        TensorShape output_shape{input_t.dimension(0), out_row_dim, out_col_dim, input_t.dimension(3)};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output));
        auto output_t = output->tensor<T, 4>();

        //using Dtype = T;
        //Dtype *col_buff = NULL;
        //if (!is_1x1_ || normalize_patches_) {
        //    col_buff = col_buffer_.mutable_cpu_data();
        //}

        // What is the interlacing stuff?
        // What is the bottom index?

        const int params_size = num_instances_ * block_w_ * block_h_ * block_c_;
        typename vec<Dtype>::vec2 *inter_params = static_cast<typename vec<Dtype>::vec2 *>(interlaced_params_->mutable_cpu_data());
        interlace_cpu<Dtype>(params_size, templates, weights, inter_params);

        TTypes<T, 4>::Tensor columns_t = Im2Col(input_t);
        T* columns_data = columns_t.data();

        int batch_size = input_t.dimension(0);
        int size_of_one_input_batch = columns_t.dimension(1) * columns_t.dimension(2) * columns_t.dimension(3) * columns_t.dimension(4);
        int size_of_one_output_batch = output_t.dimension(1) * output_t.dimension(2) * output_t.dimension(3);

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            T* columns_slice = &columns_t.data()[batch_idx * size_of_one_input_batch];

            T* output_slice = output_t.data()[batch_idx * size_of_one_output_batch];

                switch (this->layer_param_.similarity_param().similarity_function()) {
                    case SimilarityParameter_SimilarityFunction_CONVOLUTION:
                        ggemm_cpu
                        < typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                sim_linear_forward < Dtype >, ggemm_add < Dtype >, false >
                                                                                   (M_, N_, K_, inter_params, col_buff,
                                                                                           top_data +
                                                                                           top[bottom_idx]->offset(
                                                                                                   n), 0, 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L1:
                        ggemm_cpu
                        < typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                sim_l1_forward < Dtype >, ggemm_add < Dtype >, false >
                                                                               (M_, N_, K_, inter_params, col_buff,
                                                                                       top_data +
                                                                                       top[bottom_idx]->offset(
                                                                                               n), 0, 0);
                        break;
                    case SimilarityParameter_SimilarityFunction_L2:
                        if (normalization_term_) {
                            if (use_log_space_weight_param_) {
                                if (ignore_nan_input_) {
                                    ggemm_cpu
                                    < typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward < Dtype, true, true >, ggemm_add < Dtype >,
                                            false >
                                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(
                                                    n), 0, normalization_term_fudge_);
                                } else {
                                    ggemm_cpu
                                    < typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward < Dtype, true, false >, ggemm_add < Dtype >,
                                            false >
                                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(
                                                    n), 0, normalization_term_fudge_);
                                    caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                                                            top_data + top[bottom_idx]->offset(n));
                                }
                            } else {
                                if (ignore_nan_input_) {
                                    ggemm_cpu
                                    < typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward < Dtype, false, true >, ggemm_add < Dtype >,
                                            false >
                                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(
                                                    n), 0, normalization_term_fudge_);
                                } else {
                                    ggemm_cpu
                                    < typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
                                            sim_l2_normalized_forward < Dtype, false, false >, ggemm_add < Dtype >,
                                            false >
                                            (M_, N_, K_, inter_params, col_buff, top_data + top[bottom_idx]->offset(
                                                    n), 0, normalization_term_fudge_);
                                    caffe_add_scalar<Dtype>(M_ * N_, Dtype(-0.5) * Dtype(K_) * std::log(2.0 * M_PI),
                                                            top_data + top[bottom_idx]->offset(n));
                                }
                            }
                        } else {
                            if (use_log_space_weight_param_) {
                                ggemm_cpu
                                < typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                        sim_l2_forward < Dtype, true >, ggemm_add < Dtype >, false >
                                                                                             (M_, N_, K_, inter_params, col_buff,
                                                                                                     top_data +
                                                                                                     top[bottom_idx]->offset(
                                                                                                             n), 0, 0);
                            } else {
                                ggemm_cpu
                                < typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
                                        sim_l2_forward < Dtype, false >, ggemm_add < Dtype >, false >
                                                                                              (M_, N_, K_, inter_params, col_buff,
                                                                                                      top_data +
                                                                                                      top[bottom_idx]->offset(
                                                                                                              n), 0, 0);
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
    }
};





