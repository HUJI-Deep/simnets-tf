//
// Created by elhanan7 on 07/03/17.
//
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "kernels/similarity_kernel_common.hpp"

using namespace tensorflow;


namespace {

    template<typename T>
    inline T* ptr_at(typename TTypes<T, 4>::Tensor t, const int n, const int c = 0, const int h = 0,
                           const int w = 0)  {
        int offset = ((n * t.dimension(1) + c) * t.dimension(2) + h) * t.dimension(3) + w;
        return t.data() + offset;
    }

    template <typename Dtype> Dtype similarity_function(SimilarityFunction sim_func,
                                                        const Dtype x, const Dtype z, const int K) {
        switch (sim_func) {
            case SIM_FUNC_L1:
                return -std::abs(x-z);
            case SIM_FUNC_L2:
                return  -(x-z) * (x-z);
            default:
                return 0;
        }
    }
}
template<typename T>
class SimilarityRefKernelCPU : public SimilarityKernelCommon {
public:

    using Base = SimilarityKernelCommon;
    using Dtype = T;

    SimilarityRefKernelCPU(OpKernelConstruction *context) : Base(context) {}

    void Compute(OpKernelContext *context) override {
        this->CalculateDimensions<T>(context);
        auto input = context->input(0);
        auto templates = context->input(1);
        auto weights = context->input(2);

        auto input_t = input.tensor<T, 4>();
        auto templates_t = templates.tensor<T, 4>();
        auto weights_t = weights.tensor<T, 4>();

        Tensor *output = NULL;

        TensorShape output_shape{batch_, out_c_, out_h_, out_w_};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output));
        auto output_t = output->tensor<T, 4>();

        this->SimilarityRefImplementation(input_t, weights_t, templates_t, output_t);
    }

    void SimilarityRefImplementation(typename TTypes<Dtype, 4>::Tensor input_t,
                                     typename TTypes<Dtype, 4>::Tensor weights_t,
                                     typename TTypes<Dtype, 4>::Tensor templates_t,
                                     typename TTypes<Dtype, 4>::Tensor output_t) {

        output_t = output_t.constant(0);

        // Similarity
        for (int n = 0; n < batch_; n++) {
            for (int o = 0; o < num_instances_; o++) {
                for (int y = 0; y < out_h_; y++) {
                    for (int x = 0; x < out_w_; x++) {
                        for (int k = 0; k < block_c_; k++) {
                            for (int p = 0; p < block_h_; p++) {
                                for (int q = 0; q < block_w_; q++) {
                                    const Dtype u = weights_t(o, k, p, q);
                                    const Dtype z = templates_t(o, k, p, q);
                                    int in_y = y * stride_h_ - pad_h_ + p;
                                    int in_x = x * stride_w_ - pad_w_ + q;
                                    Dtype pixel = out_of_bounds_value_; // Needs to be specified! 
                                    if (in_y >= 0 && in_y < height_
                                        && in_x >= 0 && in_x < width_) {
                                        pixel = input_t(n, k, in_y, in_x);
                                    }
                                    if (ignore_nan_input_ && std::isnan(pixel)) {
                                        continue;
                                    }
                                    Dtype value = u * similarity_function(similarity_function_, pixel, z, block_h_ * block_w_);
                                    if (normalization_term_) {
                                        value *= 0.5;
                                        value += 0.5 * std::log(u + normalization_term_fudge_)
                                                 - 0.5 * std::log(2.0 * M_PI);
                                    }
                                    output_t(n, o, y, x) += value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

REGISTER_OP("SimilarityRef")
        .Input("input: T")
        .Input("templates: T")
        .Input("weights: T")
        .Output("output: T")
        .Attr("T: {float32, float64}")
        .Attr("similarity_function: {'L1', 'L2'} = 'L2'")
        .Attr("ksize: list(int) = [3,3]")
        .Attr("strides: list(int) = [2,2]")
        .Attr("padding: list(int) = [0,0]")
        .Attr("normalization_term: bool = false")
        .Attr("normalization_term_fudge: float = 0.001")
        .Attr("ignore_nan_input: bool = false")
        .Attr("out_of_bounds_value: float = 0.0");

REGISTER_KERNEL_BUILDER(
        Name("SimilarityRef")
                .Device(DEVICE_CPU)
                .TypeConstraint<float>("T"),
        SimilarityRefKernelCPU<float>);
REGISTER_KERNEL_BUILDER(
        Name("SimilarityRef")
                .Device(DEVICE_CPU)
                .TypeConstraint<double>("T"),
        SimilarityRefKernelCPU<double>);
