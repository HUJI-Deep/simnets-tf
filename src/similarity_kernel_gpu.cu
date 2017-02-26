
#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

namespace
{
    // Divide such that DivToSmall(a, b) = c ==> b*c <= a
    // this is  different from regular integer division for negative numbers
    template <typename T>
    T DivToSmaller(T a, T b)
    {
        auto q = a / b;
        // assuming round-toward-zero
        if ((a < 0) && (q * b != a)) --q;
        return q;
    }

}

template <typename T>
class SumPoolOpCPU : public OpKernel {
 public:
  explicit SumPoolOpCPU(OpKernelConstruction* context) : OpKernel(context)
  {
     OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
     OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
     OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
  }


  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T,4>();

    // Create an output tensor
    Tensor* output_tensor = NULL;

      int64 outRowDim, outColDim;
      int64 outRowPad, outColPad;
      GetWindowedOutputSize(input.dimension(1), ksize_[1], strides_[1], padding_, &outRowDim, &outRowPad);
      GetWindowedOutputSize(input.dimension(2), ksize_[2], strides_[2], padding_, &outColDim, &outColPad);

      // Somehow, the values calculated in the previous function are not compatible with what actually goes on
      // in convolutions, nor with the documentation of the function. the following code solves the problem
      if (padding_ == SAME)
      {
          outRowPad = ((outRowDim - 1) * strides_[1] + ksize_[1] - input.dimension(1)) / 2;
          outColPad = ((outColDim - 1) * strides_[2] + ksize_[2] - input.dimension(2)) / 2;
      }

      TensorShape outputShape{input.dimension(0), outRowDim, outColDim, input.dimension(3)};
      OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                     &output_tensor));
      auto output = output_tensor->tensor<T, 4>();
      SumPool(input, outRowPad, outColPad, output);

      context->set_output(0, *output_tensor);
  }

    /// Compute one output element
    /// @param batch the batch dimension
    /// @param channel the channel we are working on
    /// @param rowStart the starting row, could be negative when padding is on
    /// @param colStart the starting column, could be negative when padding is enabled
    /// @param data the input tensor with rank 4
    /// @returns the sum of all the elements in the window
    T ComputeElement(int64 batch, int64 channel, int64 rowStart, int64 colStart, const typename TTypes<T, 4>::ConstTensor& data)
    {
        T sum{0};
        int64 lowerBoundRow = std::max(int64(0), rowStart);
        int64 lowerBoundCol = std::max(int64(0), colStart);
        int64 upperBoundRow = std::min(rowStart + ksize_[1], int64(data.dimension(1)));
        int64 upperBoundCol = std::min(colStart + ksize_[2], int64(data.dimension(2)));

        for (int64 row_index = lowerBoundRow; row_index < upperBoundRow; ++row_index)
        {
            for (int64 col_index = lowerBoundCol; col_index < upperBoundCol; ++col_index)
            {
                sum += data(batch, row_index, col_index, channel);
            }
        }
        return sum;
    }

    /// Compute sum pool over the input tensor
    /// @param input the input tensor with rank 4
    /// @param padRow the row padding needed to calculate the indices
    /// @param padCol the column padding needed to calculate the indices
    /// @param output the output tensor, with rank 4
    void SumPool(typename TTypes<T, 4>::ConstTensor& input, int64 padRow, int64 padCol, typename TTypes<T, 4>::Tensor& output)
    {
        for (int64 batch = 0; batch < output.dimension(0); ++batch)
        {
            for (int64 row = 0; row < output.dimension(1); ++row)
            {
                for (int64 col = 0; col < output.dimension(2); ++col)
                {
                    for (int64 channel = 0; channel < output.dimension(3); ++channel)
                    {
                        int64 inputRowStart = row * strides_[1] - padRow;
                        int64 inputColStart = col * strides_[2] - padCol;
                        output(batch, row, col, channel) = ComputeElement(batch, channel, inputRowStart, inputColStart, input);
                    }
                }
            }
        }
    }


private:
   Padding padding_;
   std::vector<int> ksize_;
   std::vector<int> strides_;
};

template <typename T>
class SumPoolOpGradCPU : public OpKernel {
public:
    explicit SumPoolOpGradCPU(OpKernelConstruction* context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
        OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    }


    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& original_input_tensor = context->input(0);
        auto original_input = original_input_tensor.tensor<T,4>();
        const Tensor& original_output_tensor = context->input(1);
        auto original_output = original_output_tensor.tensor<T,4>();

        // Create an output tensor
        Tensor* output_tensor = NULL;

        int64 outRowDim, outColDim;
        int64 outRowPad, outColPad;
        GetWindowedOutputSize(original_input.dimension(1), ksize_[1], strides_[1], padding_, &outRowDim, &outRowPad);
        GetWindowedOutputSize(original_input.dimension(2), ksize_[2], strides_[2], padding_, &outColDim, &outColPad);

        // Somehow, the values calculated in the previous function are not compatible with what actually goes on
        // in convolutions, nor with the documentation of the function. the following code solves the problem
        if (padding_ == SAME)
        {
            outRowPad = ((outRowDim - 1) * strides_[1] + ksize_[1] - original_input.dimension(1)) / 2;
            outColPad = ((outColDim - 1) * strides_[2] + ksize_[2] - original_input.dimension(2)) / 2;
        }

        OP_REQUIRES_OK(context, context->allocate_output(0, original_input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->tensor<T, 4>();
        SumPoolGrad(original_output, outRowPad, outColPad, output);

        context->set_output(0, *output_tensor);
    }

    /// Compute one output element
    /// @param batch the batch dimension
    /// @param channel the channel we are working on
    /// @param rowStart the starting row, could be negative when padding is on
    /// @param colStart the starting column, could be negative when padding is enabled
    /// @param data the input tensor with rank 4
    /// @returns the sum of all the elements in the window
    T ComputeElement(int64 batch, int64 channel, int64 rowStart, int64 colStart, const typename TTypes<T, 4>::ConstTensor& data)
    {
        T sum{0};
        int64 lowerBoundRow = std::max(int64(0), DivToSmaller(rowStart, int64(strides_[1])) + 1);
        int64 lowerBoundCol = std::max(int64(0), DivToSmaller(colStart, int64(strides_[2])) + 1);
        int64 upperBoundRow = std::min(DivToSmaller(rowStart + ksize_[1], int64(strides_[1])), int64(data.dimension(1) - 1));
        int64 upperBoundCol = std::min(DivToSmaller(colStart + ksize_[2], int64(strides_[2])), int64(data.dimension(2) - 1));

        for (int64 row_index = lowerBoundRow; row_index <= upperBoundRow; ++row_index)
        {
            for (int64 col_index = lowerBoundCol; col_index <= upperBoundCol; ++col_index)
            {
                sum += data(batch, row_index, col_index, channel);
            }
        }
        return sum;
    }

    /// Compute sum pool over the input tensor
    /// @param original_output the original output tensor for the sum pool operation, with rank 4
    /// @param padRow the row padding needed to calculate the indices
    /// @param padCol the column padding needed to calculate the indices
    /// @param output the output tensor, with rank 4
    void SumPoolGrad(typename TTypes<T, 4>::ConstTensor& original_output, int64 padRow, int64 padCol, typename TTypes<T, 4>::Tensor& output)
    {
        for (int64 batch = 0; batch < output.dimension(0); ++batch)
        {
            for (int64 row = 0; row < output.dimension(1); ++row)
            {
                for (int64 col = 0; col < output.dimension(2); ++col)
                {
                    for (int64 channel = 0; channel < output.dimension(3); ++channel)
                    {
                        int64 inputRowStart = row + padRow - ksize_[1];
                        int64 inputColStart = col + padCol - ksize_[2];
                        output(batch, row, col, channel) = ComputeElement(batch, channel, inputRowStart, inputColStart, original_output);
                    }
                }
            }
        }
    }


private:
    Padding padding_;
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
REGISTER_KERNEL_BUILDER(
        Name("SumPoolGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<int32>("T"),
        SumPoolOpGradCPU<int>);
REGISTER_KERNEL_BUILDER(
        Name("SumPoolGrad")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        SumPoolOpGradCPU<float>);