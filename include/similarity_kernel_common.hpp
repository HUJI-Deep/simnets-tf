//
// Created by elhanani on 26/02/17.
//

#ifndef SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
#define SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h>

template <typename T>
class SimilarityKernelCommon : public tensorflow::OpKernel
{
public:
    explicit SimilarityKernelCommon(OpKernelConstruction* context);

protected:
    std::vector<int> ksize_;
    std::vector<int> stride_;
    tensorflow::Padding padding_;
    bool normalization_term_;
    T normalization_term_fudge_;
    T normalization_fudge_factor_;
    bool ignore_nan_input_;
};

template <typename T>
SimilarityKernelCommon<T>::SimilarityKernelCommon(OpKernelConstruction* context)
{
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_term", &normalization_term_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_term_fudge", &normalization_term_fudge_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_fudge_factor_", &normalization_fudge_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_nan_input", &ignore_nan_input_));
}

#endif //SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
