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
    auto input = input_tensor.tensor<T,4>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
      TensorShape outputShape;
      outputShape.AddDim(input.dimension(0));
      outputShape.AddDim(GetOutputDim(input.dimension(1), ksize_[1], strides_[1], paddingSame_));
      outputShape.AddDim(GetOutputDim(input.dimension(2), ksize_[2], strides_[2], paddingSame_));
      outputShape.AddDim(input.dimension(3));
    OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                     &output_tensor));
      auto output = output_tensor->tensor<T, 4>();
      if (!paddingSame_)
      {
          SumPoolValid(input, output);
      }
      else
      {
          SumPoolSame(input, output);
      }
     context->set_output(0, *output_tensor);
  }

    T ComputeElementValid(long batch, long channel, long row, long col, const typename TTypes<T, 4>::ConstTensor& data)
    {
        T sum{0};
        long oddEvenRow = 1 - ksize_[1] % 2;
        long oddEvenCol = 1 - ksize_[2] % 2;
        long kHalfRow = ksize_[1] / 2;
        long kHalfCol = ksize_[2] / 2;
        long lowerBoundRow = row - kHalfRow + oddEvenRow;
        long upperBoundRow = row + kHalfRow + 1;
        long lowerBoundCol = col - kHalfCol + oddEvenCol;
        long upperBoundCol = col + kHalfCol + 1;
        for (long row_index = lowerBoundRow; row_index < upperBoundRow; ++row_index)
        {
            for (long col_index = lowerBoundCol; col_index < upperBoundCol; ++col_index)
            {
                sum += data(batch, row_index, col_index, channel);
            }
        }
        return sum;
    }

    T ComputeElementSame(long batch, long channel, long row, long col, const typename TTypes<T, 4>::ConstTensor& data)
    {
        T sum{0};
        long oddEvenRow = 1 - ksize_[1] % 2;
        long oddEvenCol = 1 - ksize_[2] % 2;
        long kHalfRow = ksize_[1] / 2;
        long kHalfCol = ksize_[2] / 2;
        long lowerBoundRow = row - kHalfRow + oddEvenRow;
        lowerBoundRow = std::max(0L, lowerBoundRow);
        long upperBoundRow = row + kHalfRow + 1;
        upperBoundRow = std::min(data.dimension(1), upperBoundRow);
        long lowerBoundCol = col - kHalfCol + oddEvenCol;
        lowerBoundCol = std::max(0L, lowerBoundCol);
        long upperBoundCol = col + kHalfCol + 1;
        upperBoundCol = std::min(data.dimension(2), upperBoundCol);
        for (long row_index = lowerBoundRow; row_index < upperBoundRow; ++row_index)
        {
            for (long col_index = lowerBoundCol; col_index < upperBoundCol; ++col_index)
            {
                sum += data(batch, row_index, col_index, channel);
            }
        }
        return sum;
    }

    void SumPoolValid(typename TTypes<T, 4>::ConstTensor& input, typename TTypes<T, 4>::Tensor& output)
    {
        long validMinRow = (ksize_[1] / 2) + (ksize_[1] % 2) - 1;
        long validMinCol = (ksize_[2] / 2) + (ksize_[2] % 2) - 1;
        long validMaxRow =  input.dimension(1) - (ksize_[1] / 2);
        long validMaxCol =  input.dimension(2) - (ksize_[2] / 2);
        for (long batch = 0; batch < input.dimension(0); ++batch)
        {
            for (long row = validMinRow; row < validMaxRow; row += strides_[1])
            {
                for (long col = validMinCol; col < validMaxCol; col += strides_[2])
                {
                    for (long channel = 0; channel < input.dimension(3); ++channel)
                    {
                        long outputRow = row / strides_[1];
                        long outputCol = col / strides_[2];
                        output(batch, outputRow, outputCol, channel) = ComputeElementValid(batch, channel, row, col, input);
                    }
                }
            }
        }
    }

    void SumPoolSame(typename TTypes<T, 4>::ConstTensor& input, typename TTypes<T, 4>::Tensor& output)
    {
        long minRow = (strides_[1] / 2) - (1 - strides_[1] % 2);
        long minCol = (strides_[2] / 2) - (1 - strides_[2] % 2);
        for (long batch = 0; batch < input.dimension(0); ++batch)
        {
            for (long row = minRow; row < input.dimension(1); row += strides_[1])
            {
                for (long col = minCol; col < input.dimension(2); col += strides_[2])
                {
                    for (long channel = 0; channel < input.dimension(3); ++channel)
                    {
                        long outputRow = row / strides_[1];
                        long outputCol = col / strides_[2];
                        output(batch, outputRow, outputCol, channel) = ComputeElementSame(batch, channel, row, col, input);
                    }
                }
            }
        }
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