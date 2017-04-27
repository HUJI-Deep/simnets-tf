#include "mex_kernel_common.hpp"
#include "im2col.hpp"
#include "ggemm_cpu.hpp"

using namespace tensorflow;

MEXKernelCommon::MEXKernelCommon(tensorflow::OpKernelConstruction *context)
        : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("softmax_mode", &softmax_mode_));

    OP_REQUIRES_OK(context, context->GetAttr("num_instances", &num_instances_));
    OP_REQUIRES(context, num_instances_ > 0, errors::InvalidArgument("num_instances must be positive"));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));


    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 3,
                errors::InvalidArgument("strides should be a list with 3 elements"));

    OP_REQUIRES_OK(context, context->GetAttr("blocks", &blocks_));
    OP_REQUIRES(context, blocks_.size() == 3,
                errors::InvalidArgument("blocks should be a list with 3 elements"));

    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(context, context->GetAttr("blocks_out_of_bounds_value", &blocks_out_of_bounds_value_));
    OP_REQUIRES_OK(context, context->GetAttr("blocks_round_down", &blocks_round_down_));
    OP_REQUIRES_OK(context, context->GetAttr("use_unshared_regions", &use_unshared_regions_));

    OP_REQUIRES_OK(context, context->GetAttr("shared_offset_region", &shared_offset_region_));
    OP_REQUIRES_OK(context, context->GetAttr("unshared_offset_region", &unshared_offset_region_));
    OP_REQUIRES(context, unshared_offset_region_.size() == 3 || unshared_offset_region_.size() == 1,
                    errors::InvalidArgument("unshared_offset_region should be a list with 1 or 3 elements"));
    OP_REQUIRES(context, shared_offset_region_.size() == 3 || shared_offset_region_.size() == 1,
                    errors::InvalidArgument("shared_offset_region should be a list with 1 or 3 elements"));
}


// Parameters:
// Offsets should be [num_regions_, num_instances_, block_c_, block_h_, block_w_]
void MexDimensionsData::CalculateDimensions() {

    pad_c_ = padding_[0];
    pad_h_ = padding_[1];
    pad_w_ = padding_[2];

    stride_c_ = strides_[0];
    stride_h_ = strides_[1];
    stride_w_ = strides_[2];

    block_c_ = blocks_[0];
    block_h_ = blocks_[1];
    block_w_ = blocks_[2];

    height_out_ = simnets_tf::dimension_out_size(height_, pad_h_, block_h_, stride_h_, blocks_round_down_);
    width_out_ = simnets_tf::dimension_out_size(width_, pad_w_, block_w_, stride_w_, blocks_round_down_);
    channels_out_ = simnets_tf::dimension_out_size(channels_, pad_c_, block_c_, stride_c_, blocks_round_down_);

    if (!use_unshared_regions_) {

        if (shared_offset_region_.size() == 3) {
            shared_offsets_region_c_ = shared_offset_region_[0];
            shared_offsets_region_h_ = shared_offset_region_[1];
            shared_offsets_region_w_ = shared_offset_region_[2];
        } else {
            shared_offsets_region_w_ = shared_offsets_region_h_ = shared_offset_region_[0];
            shared_offsets_region_c_ = -1;
        }

        if (shared_offsets_region_h_ < 0) {
            shared_offsets_region_h_ = height_out_;
        }
        shared_offsets_region_h_ = std::min(shared_offsets_region_h_, height_out_);
        if (shared_offsets_region_w_ < 0) {
            shared_offsets_region_w_ = width_out_;
        }
        shared_offsets_region_w_ = std::min(shared_offsets_region_w_, width_out_);
        if (shared_offsets_region_c_ < 0) {
            shared_offsets_region_c_ = channels_out_;
        }
        shared_offsets_region_c_ = std::min(shared_offsets_region_c_, channels_out_);

        offsets_h_ = ceiled_div(height_out_, shared_offsets_region_h_);
        offsets_w_ = ceiled_div(width_out_, shared_offsets_region_w_);
        offsets_c_ = ceiled_div(channels_out_, shared_offsets_region_c_);

    } else {
        if (unshared_offset_region_.size() == 3) {
            unshared_offsets_region_c_ = unshared_offset_region_[0];
            unshared_offsets_region_h_ = unshared_offset_region_[1];
            unshared_offsets_region_w_ = unshared_offset_region_[2];
        } else {
            unshared_offsets_region_w_ = unshared_offsets_region_h_ = unshared_offset_region_[0];
            unshared_offsets_region_c_ = -1;
        }

        if (unshared_offsets_region_h_ < 0) {
            unshared_offsets_region_h_ = 1;
        }
        unshared_offsets_region_h_ = std::min(unshared_offsets_region_h_, height_out_);
        if (unshared_offsets_region_w_ < 0) {
            unshared_offsets_region_w_ = 1;
        }
        unshared_offsets_region_w_ = std::min(unshared_offsets_region_w_, width_out_);
        if (unshared_offsets_region_c_ < 0) {
            unshared_offsets_region_c_ = 1;
        }
        unshared_offsets_region_c_ = std::min(unshared_offsets_region_c_, channels_out_);
        offsets_h_ = unshared_offsets_region_h_;
        offsets_w_ = unshared_offsets_region_w_;
        offsets_c_ = unshared_offsets_region_c_;
        shared_offsets_region_h_ = ceiled_div(height_out_, unshared_offsets_region_h_);
        shared_offsets_region_w_ = ceiled_div(width_out_, unshared_offsets_region_w_);
        shared_offsets_region_c_ = ceiled_div(channels_out_, unshared_offsets_region_c_);
    }

    num_regions_ = offsets_h_ * offsets_w_ * offsets_c_;
    region_size_ = shared_offsets_region_w_ * shared_offsets_region_h_ * shared_offsets_region_c_;
    channels_out_total_ = channels_out_ * num_instances_;
    // Prepare the matrix multiplication computation.
    // Each input will be convolved as a single GEMM.
    M_ = num_instances_;
    K_ = block_c_ * block_h_ * block_w_;
    N_ = height_out_ * width_out_ * channels_out_;

    is_1x1_ = (block_c_ == channels_ || (block_c_ == 1 && stride_c_ == 1)) && block_w_ == 1 && block_h_ == 1
              && stride_h_ == 1 && stride_w_ == 1
              && pad_c_ == 0 && pad_h_ == 0 && pad_w_ == 0;
    is_2D_pooling_ = num_instances_ == 1
                     && block_c_ == 1 && stride_c_ == 1 && pad_c_ == 0;


}


void MexDimensionsData::CalculateDimensionsWithConext(tensorflow::OpKernelContext *context) {
    // Configure input size and number of instances
    batch_ = context->input(0).shape().dim_size(0);
    channels_ = context->input(0).shape().dim_size(1);
    height_ = context->input(0).shape().dim_size(2);
    width_ = context->input(0).shape().dim_size(3);

    CalculateDimensions();
    OP_REQUIRES(context, block_c_ <= channels_, errors::InvalidArgument("block depth must be smaller than input depth"));
}
