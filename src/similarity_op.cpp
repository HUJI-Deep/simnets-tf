#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
namespace si = tensorflow::shape_inference;

REGISTER_OP("Similarity")
    .Input("value: T")
    .Output("output: T")
    .Attr("T: {float32, float64}")
    .Attr("ksize: list(int) = [1,2,2,1]")
    .Attr("strides: list(int) = [1,2,2,1]")
    .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
    .SetShapeFn(si::Conv2DShape)
    .Doc(R"doc(
Performs sum pooling on the input.
Each entry in `output` is the sum of the corresponding size `ksize`
window in `value`.
value: 4-D with shape `[batch, height, width, channels]`.
ksize: The size of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
strides: The stride of the sliding window for each dimension of `value` (batch and channel dimension must be 1).
padding: The type of padding algorithm to use.
output: The sum pooled output tensor.
)doc");

REGISTER_OP("SumPoolGrad")
        .Input("original_input: T")
        .Input("incoming_gradient: T")
        .Output("output_gradient: T")
        .Attr("T: {float32, int32}")
        .Attr("ksize: list(int) = [1,2,2,1]")
        .Attr("strides: list(int) = [1,2,2,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'SAME'");
