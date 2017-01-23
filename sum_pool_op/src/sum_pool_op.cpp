#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
namespace si = tensorflow::shape_inference;

Status GetWindowedOutputSizeFromDims(
        si::InferenceContext* c,
        si::DimensionHandle input_size,
        int filter_size, int stride,
        const std::string& padding, si::DimensionHandle* output_size) {
    if (stride <= 0) {
        return errors::InvalidArgument("Stride must be > 0, but got ", stride);
    }

    if (padding == "VALID") {
        TF_RETURN_IF_ERROR(c->Subtract(input_size, filter_size, output_size));
        TF_RETURN_IF_ERROR(c->Add(*output_size, stride, output_size));
        TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                     false /* evenly_divisible */, output_size));
    }
    else if (padding == "SAME")
    {
            TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
            TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                         false /* evenly_divisible */, output_size));
    }
    return Status::OK();
}

Status MaxPoolInferShape(si::InferenceContext* c)
{
   si::ShapeHandle inputShape = c->input(0);
   si::ShapeHandle outputShape;
   TF_RETURN_IF_ERROR(c->WithRank(inputShape, 4, &outputShape));
   std::string padding;
   std::vector<int> ksize, strides;
   TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
   TF_RETURN_IF_ERROR(c->GetAttr("ksize", &ksize));
   TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
   si::DimensionHandle cols, rows;
   TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(c, c->Dim(inputShape, 1),
                                                    ksize[1], strides[1], padding, &rows));
   TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(c, c->Dim(inputShape, 2),
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
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn(MaxPoolInferShape);
