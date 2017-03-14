//
// Created by elhanani on 26/02/17.
//

#ifndef SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
#define SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"


enum SimilarityFunction {
    SIM_FUNC_L1,
    SIM_FUNC_L2,
};

class SimilarityKernelCommon : public tensorflow::OpKernel {
public:
    explicit SimilarityKernelCommon(tensorflow::OpKernelConstruction *context);

protected:
    std::vector<int> ksize_;
    std::vector<int> stride_;
    tensorflow::Padding padding_;
    bool normalization_term_;
    float normalization_term_fudge_;
    bool ignore_nan_input_;
    float out_of_bounds_value_;
    SimilarityFunction similarity_function_;

    template <typename  T>
    void CalculateDimensions(tensorflow::OpKernelContext *context);

    long num_instances_;

    const int BATCH_DIM = 0;
    const int C_DIM = 1;
    const int H_DIM = 2;
    const int W_DIM = 3;

    long batch_;
    long height_;
    long width_;
    long channels_;


    int block_h_;
    int block_w_;
    int block_c_;

    int stride_h_;
    int stride_w_;
    int stride_c_;

    tensorflow::int64 out_h_;
    tensorflow::int64 out_w_;
    tensorflow::int64 out_c_;

    tensorflow::int64 pad_h_;
    tensorflow::int64 pad_w_;
    tensorflow::int64 pad_c_;
    bool is_1x1_;

    long M_;
    long K_;
    long N_;
};

inline
SimilarityKernelCommon::SimilarityKernelCommon(tensorflow::OpKernelConstruction *context)
        : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    std::string similarity_func_str;
    OP_REQUIRES_OK(context, context->GetAttr("similarity_function", &similarity_func_str));
    if (similarity_func_str == "L1") {
        similarity_function_ = SIM_FUNC_L1;
    } else if (similarity_func_str == "L2") {
        similarity_function_ = SIM_FUNC_L2;
    } else {
        assert(false); // Bad similarity function
    }

    OP_REQUIRES_OK(context, context->GetAttr("normalization_term", &normalization_term_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_term_fudge", &normalization_term_fudge_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_nan_input", &ignore_nan_input_));
    OP_REQUIRES_OK(context, context->GetAttr("out_of_bounds_value", &out_of_bounds_value_));
}

template <typename T>
inline
void SimilarityKernelCommon::CalculateDimensions(tensorflow::OpKernelContext *context)
{
    auto input = context->input(0);
    auto templates = context->input(1);
    auto weights = context->input(2);

    auto input_t = input.tensor<T, 4>();
    auto templates_t = templates.tensor<T, 4>();
    auto weights_t = weights.tensor<T, 4>();

    out_of_bounds_value_ = 0;

    num_instances_ = templates_t.dimension(0);

    batch_ = input_t.dimension(BATCH_DIM);
    height_ = input_t.dimension(H_DIM);
    width_ = input_t.dimension(W_DIM);
    channels_ = input_t.dimension(C_DIM);


    block_h_ = this->ksize_[1];
    block_w_ = this->ksize_[2];
    block_c_ = channels_;

    stride_h_ = this->stride_[1];
    stride_w_ = this->stride_[2];
    stride_c_ = channels_;

    GetWindowedOutputSize(height_, block_h_, stride_h_, this->padding_, &out_h_, &pad_h_);
    GetWindowedOutputSize(width_,  block_w_, stride_w_, this->padding_, &out_w_, &pad_w_);

    out_c_ = num_instances_;
    pad_c_ = 0;
    is_1x1_ = block_c_ == channels_ && block_w_ == 1 && block_h_ == 1
                  && stride_h_ == 1 && stride_w_ == 1
                  && pad_c_ == 0 && pad_h_ == 0 && pad_w_ == 0;

    M_ = num_instances_;
    K_ = block_c_ * block_h_ * block_w_;
    N_ = out_h_ * out_w_;

    OP_REQUIRES(context, channels_ == templates_t.dimension(1),
                tensorflow::errors::InvalidArgument("Number of channels mismatch between input and template"));
    OP_REQUIRES(context, channels_ == weights_t.dimension(1),
                tensorflow::errors::InvalidArgument("Number of channels mismatch between input and weights"));
}

#endif //SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
