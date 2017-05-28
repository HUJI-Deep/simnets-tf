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
        .SetShapeFn(MexShape);
//      .Doc(R"doc(
// Performs sum pooling on the input.
// Each entry in `output` is the sum of the corresponding size `ksize`
// window in `value`.
// value: 4-D with shape `[batch, height, width, channels]`.
// ksize: The size of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
// strides: The stride of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
// padding: The type of padding algorithm to use.
// output: The sum pooled output tensor.
// )doc");

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