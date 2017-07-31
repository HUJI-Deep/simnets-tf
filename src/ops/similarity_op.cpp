#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "utils/im2col.hpp"

using namespace tensorflow;

Status SimilarityShape(shape_inference::InferenceContext* c) {
    using namespace tensorflow::shape_inference;
    ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
    ShapeHandle template_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &template_shape));

    std::vector<int32> strides;
    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
    std::vector<int32> blocks;
    TF_RETURN_IF_ERROR(c->GetAttr("blocks", &blocks));
    std::vector<int32> padding;
    TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

    if (strides.size() != 2) {
        return errors::InvalidArgument(
                "Similarity requires the stride attribute to contain 2 values, but got: ",
                strides.size());
    }

    int32 stride_rows = strides[0];
    int32 stride_cols = strides[1];

    DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
    DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
    DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
    DimensionHandle filter_rows_dim = c->Dim(template_shape, 2);
    DimensionHandle filter_cols_dim = c->Dim(template_shape, 3);
    DimensionHandle output_depth_dim = c->Dim(template_shape, 0);

    DimensionHandle unused;
    TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(input_shape, 1), c->Dim(template_shape, 1), &unused));

    DimensionHandle output_rows, output_cols;
    TF_RETURN_IF_ERROR(simnets_tf::GetSimnetsOutputSizeFromDims(
            c, in_rows_dim, filter_rows_dim, stride_rows, padding[0], true, &output_rows));
    TF_RETURN_IF_ERROR(simnets_tf::GetSimnetsOutputSizeFromDims(
            c, in_cols_dim, filter_cols_dim, stride_cols, padding[1], true, &output_cols));

    ShapeHandle output_shape;
    output_shape = c->MakeShape({batch_size_dim, output_depth_dim, output_rows, output_cols});

    c->set_output(0, output_shape);
    return Status::OK();
}

Status SimilarityParametersGradShape(shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1));
    c->set_output(1, c->input(2));
    return Status::OK();
}

REGISTER_OP("Similarity")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("blocks: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityShape)
        .Doc(R"doc(
Computes a 2-D similarity given 4-D `input` `templates` and `weights` tensors.
As defined in `https://arxiv.org/abs/1506.03059`
Given an input tensor of shape `[batch, in_channels, in_height, in_width]`
and a templates, weights tensor of shape
`[out_channels, in_channels, filter_height, filter_width]`, this op
performs the following:
1. Extract virtual patches of size `blocks` from the input tensor,
   according to the `padding`, `strides` and `blocks` parameters.
   block size in the channels dimension is always the number of input channels.
   this results in a 2D grid of patches indexed by i,j
2. For the simplest version, for output element e = `[b, c, i, j]`, compute
   output[b, c, i ,j] = sum(weights[c] * phi(templates[c], patches[i, j]))
   where phi is either -|a - b|_1 (l1) or -|a - b|_2 (l2)
In detail:
    output[b, c, i, j] =
        sum_{dc, di, dj} templates[c, dc, di, dj] *
                           phi(input[b, dc, strides[0] * i + di - padding[0],
                                            strides[1] * j + dj - padding[1]], templates[c, dc, di, dj]
input: A 4-D tensor. with dimensions `[batch, in_channels, in_height, in_width]`.
templates: A 4-D tensor of shape
    `[out_channels, in_channels, filter_height, filter_width]`
weights: A 4-D tensor of shape
    `[out_channels, in_channels, filter_height, filter_width]`
    must be non negative!
output: A 4-D tensor of shape
    `[batch, out_channels, out_height, out_width]`
blocks: 1-D tensor of length 2.  The height and width of the blocks.
strides: 1-D tensor of length 2.  The stride of the sliding window
    for the height and width dimension of `input`.
padding: 1-D tensor of length 2.  The padding to use
    for the height and width dimension of `input`.
normalization_term:
normalization_term_fudge:
ignore_nan_input:
out_of_bounds_value:
)doc");


REGISTER_OP("SimilarityRef")
.Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("blocks: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityShape);

REGISTER_OP("SimilarityInputGrad")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Input("input_grad: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("blocks: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SimilarityParametersGrad")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Input("output_grad: T")
        .Output("templates_grad: T")
        .Output("weights_grad: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("blocks: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityParametersGradShape);

