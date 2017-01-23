#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

//Status GetWindowedOutputSizeFromDims(
//        shape_inference::InferenceContext* c,
//        shape_inference::DimensionHandle input_size,
//        shape_inference::DimensionOrConstant filter_size, int64 stride,
//        const std::string& padding, shape_inference::DimensionHandle* output_size) {
//    if (stride <= 0) {
//        return errors::InvalidArgument("Stride must be > 0, but got ", stride);
//    }
//
//    if (padding == "VALID") {
//        TF_RETURN_IF_ERROR(c->Subtract(input_size, filter_size, output_size));
//        TF_RETURN_IF_ERROR(c->Add(*output_size, stride, output_size));
//        TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
//                                     false /* evenly_divisible */, output_size));
//    }
//    else if (padding == "SAME")
//    {
//            TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
//            TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
//                                         false /* evenly_divisible */, output_size));
//    }
//    return Status::OK();
//}

REGISTER_OP("SumPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("T: {float, int32}")
    .Attr("ksize: list(int) = [1,2,2,1]")
    .Attr("strides: list(int) = [1,2,2,1]")
    .Attr("padding: {'SAME', 'VALID'}");
    //.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //    c->set_output(0, c->input(0));
    //    std::string padding;
    //    c->GetAttr("padding", &padding);
    //  return Status::OK();
    //});
