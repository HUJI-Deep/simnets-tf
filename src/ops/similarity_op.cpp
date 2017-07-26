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
    std::vector<int32> ksize;
    TF_RETURN_IF_ERROR(c->GetAttr("ksize", &ksize));
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
        .Attr("ksize: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityShape);
// TODO: Address channels_first in documentation
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

REGISTER_OP("SimilarityRef")
.Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("ksize: list(int) = [3,3]")
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
        .Attr("ksize: list(int) = [3,3]")
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
        .Attr("ksize: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityParametersGradShape);

