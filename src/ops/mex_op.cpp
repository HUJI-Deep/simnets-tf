#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "utils/im2col.hpp"

using namespace tensorflow;

Status MexShape(shape_inference::InferenceContext* c) {
    using namespace tensorflow::shape_inference;
    ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    std::vector<int32> strides;
    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
    std::vector<int32> blocks;
    TF_RETURN_IF_ERROR(c->GetAttr("blocks", &blocks));
    std::vector<int32> padding;
    TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

    if (strides.size() != 3) {
        return errors::InvalidArgument(
                "Mex requires the stride attribute to contain 3 values, but got: ",
                strides.size());
    }

    int num_instances;
    TF_RETURN_IF_ERROR(c->GetAttr("num_instances", &num_instances));
    bool blocks_round_down;
    TF_RETURN_IF_ERROR(c->GetAttr("blocks_round_down", &blocks_round_down));


    int32 stride_channels = strides[0];
    int32 stride_rows = strides[1];
    int32 stride_cols = strides[2];

    DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
    DimensionHandle in_channels_dim = c->Dim(input_shape, 1);
    DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
    DimensionHandle in_cols_dim = c->Dim(input_shape, 3);

    DimensionHandle output_channels, output_rows, output_cols;
    TF_RETURN_IF_ERROR(simnets_tf::GetSimnetsOutputSizeFromDims(
            c, in_channels_dim, blocks[0], stride_channels, padding[0], blocks_round_down, &output_channels));
    TF_RETURN_IF_ERROR(c->Multiply(output_channels, num_instances, &output_channels));
    TF_RETURN_IF_ERROR(simnets_tf::GetSimnetsOutputSizeFromDims(
            c, in_rows_dim, blocks[1], stride_rows, padding[1], blocks_round_down, &output_rows));
    TF_RETURN_IF_ERROR(simnets_tf::GetSimnetsOutputSizeFromDims(
            c, in_cols_dim, blocks[2], stride_cols, padding[2], blocks_round_down, &output_cols));

    ShapeHandle output_shape;
    output_shape = c->MakeShape({batch_size_dim, output_channels, output_rows, output_cols});

    c->set_output(0, output_shape);
    return Status::OK();
}


REGISTER_OP("Mex")
.Input("input: T")
        .Input("offsets: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("num_instances: int")
        .Attr("softmax_mode: bool = false")
        .Attr("padding: list(int) = [0, 0, 0]")
        .Attr("strides: list(int) = [1, 1, 1]")
        .Attr("blocks: list(int) = [1, 1, 1]")
        .Attr("epsilon: float = 1.0")
        .Attr("blocks_out_of_bounds_value: float = 0.0")
        .Attr("blocks_round_down: bool = true")
        .Attr("use_unshared_regions: bool = true")
        .Attr("shared_offset_region: list(int) = [-1]")
        .Attr("unshared_offset_region: list(int) = [-1]")
        .SetShapeFn(MexShape)
// TODO: Address channels_first in documentation
        .Doc(R"doc(
Computes the MEX layer given 4-D `input` and 5-D `offsets` tensors.

As defined in https://arxiv.org/abs/1506.03059

Given an input tensor of shape `[batch, in_channels, in_height, in_width]`
and a offsets tensor of shape
`[num_regions, num_instances, filter_channels, filter_height, filter_width]`,  where
num_regions is calculated from the output dimensions and the shared/unshared offsets parmaeter

This op performs the following:
Extract virtual patches of size `blocks` from the input tensor,
according to the `padding`, `strides` and `blocks` parameters.
this results in a 3D grid of patches indexed by c,i,j.
For each output element we select the corresponding patch and offsets region
then calculate:

.. math:: \frac{1}{\epsilon} \log\left(\frac{1}{n} \sum\exp(\epsilon (patch + region))\right)

The different parameters change the behaviour as described below.

input: A 4-D tensor. with dimensions `[batch, in_channels, in_height, in_width]`.
offsets: A 5-D tensor of shape
    `[num_regions, num_instances, filter_channels, filter_height, filter_width]`
    must be non negative!
output: A 4-D tensor of shape
    `[batch, out_channels, out_height, out_width]`
num_instances: the number of instances of the layer.
softmax_mode: in softmax mode we do not divide by the patch size inside of the log
blocks: list of length 3.  The 3D dimensions of the blocks.
strides: list of length 3.  The stride of the sliding window
    for the dimensions of `input`.
padding: list of length 3.  The padding to use
    for the dimensions of `input`.
epsilon: the epsilon parameter. can be +inf, -inf
blocks_out_of_bounds_value: value to use for out of bounds elements
blocks_round_down: controls the calculation of the output size.
                with round_down it is::

                    image_size + 2 * pad_size - patch_size) / stride + 1

                without it is::

                    static_cast<int>(
                       std::ceil(static_cast<float>(
                           image_size + 2 * pad_size - patch_size) / stride)) + 1

use_unshared_regions: alternative to defining a shared region, unshared region.
shared_offset_region: the region in which offsets are shared.
                    a value of -1 is replaced by the entire respective dimension.
                    can be a list of length 3, or 1. if it is of length 1 [d], it is
                    expanded to [-1, d, d]
unshared_offset_region: the region in which offsets are unshared.
                        a value of -1 is replaced by the entire respective dimension.
                        can be a list of length 3, or 1. if it is of length 1 [d], it is
                        expanded to [-1, d, d]
)doc");

REGISTER_OP("MexInputGrad")
        .Input("input: T")
        .Input("offsets: T")
        .Input("orig_output: T")
        .Input("output_grad: T")
        .Output("input_grad: T")
        .Attr("T: {float32, float64}")
        .Attr("num_instances: int")
        .Attr("softmax_mode: bool = false")
        .Attr("padding: list(int) = [0, 0, 0]")
        .Attr("strides: list(int) = [1, 1, 1]")
        .Attr("blocks: list(int) = [1, 1, 1]")
        .Attr("epsilon: float = 1.0")
        .Attr("blocks_out_of_bounds_value: float = 0.0")
        .Attr("blocks_round_down: bool = true")
        .Attr("use_unshared_regions: bool = true")
        .Attr("shared_offset_region: list(int) = [-1]")
        .Attr("unshared_offset_region: list(int) = [-1]");

REGISTER_OP("MexOffsetsGrad")
        .Input("input: T")
        .Input("offsets: T")
        .Input("orig_output: T")
        .Input("output_grad: T")
        .Output("offsets_grad: T")
        .Attr("T: {float32, float64}")
        .Attr("num_instances: int")
        .Attr("softmax_mode: bool = false")
        .Attr("padding: list(int) = [0, 0, 0]")
        .Attr("strides: list(int) = [1, 1, 1]")
        .Attr("blocks: list(int) = [1, 1, 1]")
        .Attr("epsilon: float = 1.0")
        .Attr("blocks_out_of_bounds_value: float = 0.0")
        .Attr("blocks_round_down: bool = true")
        .Attr("use_unshared_regions: bool = true")
        .Attr("shared_offset_region: list(int) = [-1]")
        .Attr("unshared_offset_region: list(int) = [-1]");

REGISTER_OP("MexRef")
.Input("input: T")
        .Input("offsets: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("num_instances: int")
        .Attr("softmax_mode: bool = false")
        .Attr("padding: list(int) = [0, 0, 0]")
        .Attr("strides: list(int) = [1, 1, 1]")
        .Attr("blocks: list(int) = [1, 1, 1]")
        .Attr("epsilon: float = 1.0")
        .Attr("blocks_out_of_bounds_value: float = 0.0")
        .Attr("blocks_round_down: bool = true")
        .Attr("use_unshared_regions: bool = true")
        .Attr("shared_offset_region: list(int) = [-1]")
        .Attr("unshared_offset_region: list(int) = [-1]");
