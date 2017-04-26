#ifndef SIMNETS_TF_MEX_LAYER_COMMON_HPP
#define SIMNETS_TF_MEX_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "im2col.hpp"


struct MexDimensionsData
{
    std::vector<int> padding_, strides_, blocks_, shared_offset_region_, unshared_offset_region_;
    void CalculateDimensionsWithConext(tensorflow::OpKernelContext* context);
    void CalculateDimensions();
    int block_c_, block_h_, block_w_;
    int stride_c_, stride_h_, stride_w_;
    int pad_c_, pad_h_, pad_w_;
    int batch_, channels_, height_, width_;
    int num_instances_;
    int channels_out_, height_out_, width_out_;
    int channels_out_total_;
    float blocks_out_of_bounds_value_;
    bool blocks_round_down_;
    bool use_log_space_parameters_;
    float linear_space_min_value_;
    int shared_offsets_region_c_, shared_offsets_region_h_, shared_offsets_region_w_;
    bool use_unshared_regions_;
    int unshared_offsets_region_c_, unshared_offsets_region_h_, unshared_offsets_region_w_;
    int offsets_c_, offsets_h_, offsets_w_;
    int region_size_;
    int num_regions_;
    bool is_1x1_;
    bool is_2D_pooling_;
    bool softmax_mode_;
    float epsilon_;

    /// M_ is the channel dimension of the output for a single group, which is the
    /// leading dimension of the filter matrix.
    int M_;
    /// K_ is the dimension of an unrolled input for a single group, which is the
    /// leading dimension of the data matrix.
    int K_;
    /// N_ is the spatial dimension of the output, the H x W, which are the last
    /// dimensions of the data and filter matrices.
    int N_;
};
class MEXKernelCommon : public tensorflow::OpKernel, protected MexDimensionsData {
public:
    MEXKernelCommon(tensorflow::OpKernelConstruction* context);
};

#endif  // SIMNETS_TF_MEX_LAYER_COMMON_HPP