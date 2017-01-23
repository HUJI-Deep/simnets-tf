#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

int GetOutputDim(int dim, long ksize, long stride, bool paddingSame)
{
    if (paddingSame)
    {
        return (dim + stride - 1) / stride;
    }
    else
    {
        return (dim - ksize + stride) / stride;
    }
}

template <typename T>
class SumPoolOpCPU : public OpKernel {
 public:
  explicit SumPoolOpCPU(OpKernelConstruction* context) : OpKernel(context)
  {
     std::string padding;
     OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
     paddingSame_ = (padding == "SAME");
     OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
     OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const auto& input = input_tensor.tensor<T,4>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
      TensorShape outputShape;
      outputShape.AddDim(input.dimension(0));
      outputShape.AddDim(GetOutputDim(input.dimension(1), ksize_[1], strides_[1], paddingSame_));
      outputShape.AddDim(GetOutputDim(input.dimension(2), ksize_[2], strides_[2], paddingSame_));
      outputShape.AddDim(input.dimension(3));
    OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                     &output_tensor));
    std::cout << output_tensor->shape().DebugString() << std::endl;
     context->set_output(0, *output_tensor);
  }
private:
   bool paddingSame_;
   std::vector<int> ksize_;
   std::vector<int> strides_;
};

REGISTER_KERNEL_BUILDER(
        Name("SumPool")
        .Device(DEVICE_CPU)
        .TypeConstraint<int32>("T"),
        SumPoolOpCPU<int>);
REGISTER_KERNEL_BUILDER(
        Name("SumPool")
        .Device(DEVICE_CPU)
        .TypeConstraint<float>("T"),
        SumPoolOpCPU<float>);