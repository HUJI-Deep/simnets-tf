#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

Status GetSimilarityOutputSizeFromDims(
        shape_inference::InferenceContext* c,
        shape_inference::DimensionHandle input_size,
        shape_inference::DimensionOrConstant filter_size, int64 stride,
        int padding, shape_inference::DimensionHandle* output_size) {
    if (stride <= 0) {
        return errors::InvalidArgument("Stride must be > 0, but got ", stride);
    }

    // See also the parallel implementation in GetWindowedOutputSizeVerbose.
    TF_RETURN_IF_ERROR(c->Add(input_size, 2 * padding, output_size));
    TF_RETURN_IF_ERROR(c->Subtract(*output_size, filter_size, output_size));
    TF_RETURN_IF_ERROR(c->Divide(*output_size, stride, false, output_size));
    TF_RETURN_IF_ERROR(c->Add(*output_size, 1, output_size));

    return Status::OK();
}

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

    if (strides.size() != 4) {
        return errors::InvalidArgument(
                "Similarity requires the stride attribute to contain 4 values, but got: ",
                strides.size());
    }

    int32 stride_rows = strides[1];
    int32 stride_cols = strides[2];

    DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
    DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
    DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
    DimensionHandle filter_rows_dim = c->Dim(template_shape, 2);
    DimensionHandle filter_cols_dim = c->Dim(template_shape, 3);
    DimensionHandle output_depth_dim = c->Dim(template_shape, 0);

    DimensionHandle unused;
    TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(input_shape, 1), c->Dim(template_shape, 1), &unused));

    Padding padding;
    TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

    int pad_h, pad_w;
    if (padding == tensorflow::VALID)
    {
        pad_h = pad_w = 0;
    } else {
        pad_h = ksize[1] / 2;
        pad_w = ksize[2] / 2;
    }

    DimensionHandle output_rows, output_cols;
    TF_RETURN_IF_ERROR(GetSimilarityOutputSizeFromDims(
            c, in_rows_dim, filter_rows_dim, stride_rows, pad_h, &output_rows));
    TF_RETURN_IF_ERROR(GetSimilarityOutputSizeFromDims(
            c, in_cols_dim, filter_cols_dim, stride_cols, pad_w, &output_cols));

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
        .Attr("ksize: list(int) = [1,2,2,1]")
        .Attr("strides: list(int) = [1,2,2,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityShape);
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

REGISTER_OP("SimilarityInputGrad")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Input("input_grad: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("ksize: list(int) = [1,2,2,1]")
        .Attr("strides: list(int) = [1,2,2,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
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
        .Attr("ksize: list(int) = [1,2,2,1]")
        .Attr("strides: list(int) = [1,2,2,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0")
        .SetShapeFn(SimilarityParametersGradShape);

