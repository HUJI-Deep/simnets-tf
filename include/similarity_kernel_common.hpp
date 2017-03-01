//
// Created by elhanani on 26/02/17.
//

#ifndef SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
#define SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"


enum SimilarityFunction
{
    SIM_FUNC_L1,
    SIM_FUNC_L2,
    SIM_FUNC_CONVOLUTION
};

template <typename T>
class SimilarityKernelCommon : public tensorflow::OpKernel
{
public:
    explicit SimilarityKernelCommon(tensorflow::OpKernelConstruction* context);

protected:
    std::vector<int> ksize_;
    std::vector<int> stride_;
    tensorflow::Padding padding_;
    bool normalization_term_;
    T normalization_term_fudge_;
    T normalization_fudge_factor_;
    bool ignore_nan_input_;
    SimilarityFunction similarity_function_;
};

template <typename T>
SimilarityKernelCommon<T>::SimilarityKernelCommon(tensorflow::OpKernelConstruction* context)
{
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    std::string similarity_func_str;
    OP_REQUIRES_OK(context, context->GetAttr("similarity_function", &similarity_func_str));
    if (similarity_func_str == "L1") {
        similarity_function_ = SIM_FUNC_L1;
    }
    else if (similarity_func_str == "L2") {
        similarity_function_ = SIM_FUNC_L2;
    }
    else if (similarity_func_str == "CONVOLUTION") {
        similarity_function_ = SIM_FUNC_CONVOLUTION;
    }
    else {
        assert(false); // Bad similarity function
    }

    OP_REQUIRES_OK(context, context->GetAttr("normalization_term", &normalization_term_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_term_fudge", &normalization_term_fudge_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_fudge_factor_", &normalization_fudge_factor_));
    OP_REQUIRES_OK(context, context->GetAttr("ignore_nan_input", &ignore_nan_input_));
}

#endif //SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
