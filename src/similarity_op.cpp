#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
namespace si = tensorflow::shape_inference;

REGISTER_OP("Similarity")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2', 'CONVOLUTION'} = 'L2'")
        .Attr("ksize: list(int) = [1,2,2,1]")
        .Attr("strides: list(int) = [1,2,2,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: T = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .SetShapeFn(si::Conv2DShape);
//         .Doc(R"doc(
// Performs sum pooling on the input.
// Each entry in `output` is the sum of the corresponding size `ksize`
// window in `value`.
// value: 4-D with shape `[batch, height, width, channels]`.
// ksize: The size of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
// strides: The stride of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
// padding: The type of padding algorithm to use.
// output: The sum pooled output tensor.
// )doc");