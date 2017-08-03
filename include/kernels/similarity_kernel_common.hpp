//
// Created by elhanani on 26/02/17.
//

#ifndef SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP
#define SIMNETS_TF_SIMILARITY_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "utils/im2col.hpp"


enum SimilarityFunction {
    SIM_FUNC_L1,
    SIM_FUNC_L2,
};

class SimilarityKernelCommon : public tensorflow::OpKernel {
public:
    /// Constructor
    /// \param context tensorflow context
    explicit SimilarityKernelCommon(tensorflow::OpKernelConstruction *context);

protected:
    /// @name patches specification
    /// output value at (i,j) corresponds to the block:<BR/>
    ///    [i * strides[0] - padding_[0], [i * strides[0] - padding_[0] + blocks[0]]
    ///       x [j * strides[1] - padding_[1], [i * strides[1] - padding_[1] + blocks[1]]
    ///@{
    std::vector<int> blocks_; //!< size of a block [h, w]
    std::vector<int> stride_; //!< strides [stride_h, stride_w]
    std::vector<int> padding_; //!< padding in each dimension [padding_h, padding_w]
    ///@}
    bool normalization_term_; //!< add normalization term, to make the operation calculate a real probability function
    float normalization_term_fudge_; //!< fudge factor for normalization
    bool ignore_nan_input_; //!< if true, nan input is ignored (marginalized)
    float out_of_bounds_value_; //!< value to use when calculation involves out of bound values
    SimilarityFunction similarity_function_; //!< type of similarity function {SIM_FUNC_L1, SIM_FUNC_L2}

    /// Calculate all the dimension data for use in the different kernels
    /// \tparam T float or double
    /// \param context the tensorflow context, holds info about actual dimensions
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
    bool is_1x1_; //!< if true the operator would use an optimized procedure

    long M_;
    long K_;
    long N_;
};

inline
SimilarityKernelCommon::SimilarityKernelCommon(tensorflow::OpKernelConstruction *context)
        : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, padding_.size() == 2,
                tensorflow::errors::InvalidArgument("padding must be a list with two elements"));
    OP_REQUIRES_OK(context, context->GetAttr("blocks", &blocks_));
    OP_REQUIRES(context, blocks_.size() == 2,
                tensorflow::errors::InvalidArgument("blocks must be a list with two elements"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 2,
                tensorflow::errors::InvalidArgument("stride must be a list with two elements"));
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
    OP_REQUIRES(context, !normalization_term_ || (similarity_function_ == SIM_FUNC_L2),
                tensorflow::errors::InvalidArgument("normalization_term is relevant only if similarity_function is L2"));
    OP_REQUIRES(context, !ignore_nan_input_ || normalization_term_,
                tensorflow::errors::InvalidArgument("ignore_nan is relevant only if normalization_term is true"));
    OP_REQUIRES_OK(context, context->GetAttr("out_of_bounds_value", &out_of_bounds_value_));
}

template <typename T>
inline
void SimilarityKernelCommon::CalculateDimensions(tensorflow::OpKernelContext *context)
{
    auto input = context->input(0);
    auto templates = context->input(1);
    auto weights = context->input(2);

    // get Eigen tensors from the tf ones
    auto input_t = input.tensor<T, 4>();
    auto templates_t = templates.tensor<T, 4>();
    auto weights_t = weights.tensor<T, 4>();

    // prepare all variables to be used in the kernels
    // the only actual calculation here is that of the output dimensions
    num_instances_ = templates_t.dimension(0);

    batch_ = input_t.dimension(BATCH_DIM);
    height_ = input_t.dimension(H_DIM);
    width_ = input_t.dimension(W_DIM);
    channels_ = input_t.dimension(C_DIM);


    block_h_ = this->blocks_[0];
    block_w_ = this->blocks_[1];
    block_c_ = channels_;

    stride_h_ = this->stride_[0];
    stride_w_ = this->stride_[1];
    stride_c_ = channels_;

    pad_h_ = padding_[0];
    pad_w_ = padding_[1];

    out_h_ = simnets_tf::dimension_out_size(height_, pad_h_, blocks_[0], stride_[0], true);
    out_w_ = simnets_tf::dimension_out_size(width_, pad_w_, blocks_[1], stride_[1], true);

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
