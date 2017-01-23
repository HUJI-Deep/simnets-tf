#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

int GetOutputDim(int dim, int ksize, int stride, bool paddingSame)
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

class SumPoolOpCPU : public OpKernel {
 public:
  explicit SumPoolOpCPU(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context->)
      TensorShape outputShape;
      outputShape.AddDim(input.dimension(0));
      outputShape.AddDim(GetOutputDim(input.dimension(1), ksize[1], strides[1], paddingSame));
      outputShape.AddDim(GetOutputDim(input.dimension(2), ksize[2], strides[2], paddingSame));
      outputShape.AddDim(input.dimension(3));
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};
