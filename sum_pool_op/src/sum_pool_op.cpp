#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
namespace si = tensorflow::shape_inference;

// Shape inference. Use the inference context to calculate the output
// shape the best we can given the input shape as known at graph creation time.
Status SumPoolInferShape(si::InferenceContext* c)
{
   si::ShapeHandle inputShape = c->input(0);
   si::ShapeHandle outputShape;
   TF_RETURN_IF_ERROR(c->WithRank(inputShape, 4, &outputShape));
   Padding padding;
   std::vector<int> ksize, strides;
   TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
   TF_RETURN_IF_ERROR(c->GetAttr("ksize", &ksize));
   TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
   si::DimensionHandle cols, rows;
   TF_RETURN_IF_ERROR(si::GetWindowedOutputSizeFromDims(c, c->Dim(inputShape, 1),
                                                    ksize[1], strides[1], padding, &rows));
   TF_RETURN_IF_ERROR(si::GetWindowedOutputSizeFromDims(c, c->Dim(inputShape, 2),
                                                    ksize[2], strides[2], padding, &cols));
   TF_RETURN_IF_ERROR(c->ReplaceDim(outputShape, 1, rows, &outputShape));
   TF_RETURN_IF_ERROR(c->ReplaceDim(outputShape, 2, cols, &outputShape));
   c->set_output(0, outputShape);
   return Status::OK();
}

REGISTER_OP("SumPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("T: {float32, int32}")
    .Attr("ksize: list(int) = [1,2,2,1]")
    .Attr("strides: list(int) = [1,2,2,1]")
    .Attr("padding: {'SAME', 'VALID'} = 'SAME'")
    .SetShapeFn(SumPoolInferShape)
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
