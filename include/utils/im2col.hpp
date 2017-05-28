//
// Created by elhanani on 01/03/17.
//

#ifndef SIMNETS_TF_IM2COL_HPP
#define SIMNETS_TF_IM2COL_HPP


#include <cmath>
#include "tensorflow/core/framework/shape_inference.h"

namespace simnets_tf {

/**
 * @brief A 3D variation of im2col. Arranges 3D patches in column matrix.
 * @details A 3D variation of im2col. Arrange 3D patches in column matrix.
 *          More specifically, it actually creates a 4D array where the 3D indexes
 *          specify where the patch originated from the original image, and the
 *          forth dimension specifies the patch data.
 *
 * @param data_im The incoming image array with multiple channels.
 * @param channels Number of channels of the input image.
 * @param height The height of the input image.
 * @param width The width of the input image.
 * @param patch_c The size of the patch in the channels dimension.
 * @param patch_h The height of the patch.
 * @param patch_w The width of the patch.
 * @param pad_c The padding along the channels dimension.
 * @param pad_h The padding along the vertical dimension.
 * @param pad_w The padding along the horizontal dimension.
 * @param stride_c The stride along the channels dimension.
 * @param stride_h The stride along the vertical dimension.
 * @param stride_w The stride along the horizontal dimension.
 * @param data_col The output column matrix.
 */
    template <typename Dtype>
    void im2col_3d_cpu(const Dtype* data_im,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_col,
                       const bool round_down = true, const Dtype out_of_bounds_value = 0);
    template <typename Dtype>
    void im2col_3d_gpu(const Dtype* data_im,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_col,
                       const bool round_down = true, const Dtype out_of_bounds_value = 0);
/**
 * @brief A 3D variation of col2im. Sums of the column matrix back to original image.
 * @details A 3D variation of col2im. Sums of the column matrix back to original image.
 *          More specifically, it actually reads from a 4D array where the 3D indexes
 *          specify where the patch originated from the original image, and the
 *          forth dimension specifies the patch data.
 *
 * @param data_col The input column matrix.
 * @param channels Number of channels of the input image.
 * @param height The height of the input image.
 * @param width The width of the input image.
 * @param patch_c The size of the patch in the channels dimension.
 * @param patch_h The height of the patch.
 * @param patch_w The width of the patch.
 * @param pad_c The padding along the channels dimension.
 * @param pad_h The padding along the vertical dimension.
 * @param pad_w The padding along the horizontal dimension.
 * @param stride_c The stride along the channels dimension.
 * @param stride_h The stride along the vertical dimension.
 * @param stride_w The stride along the horizontal dimension.
 * @param data_im The output matrix.
 */
    template <typename Dtype>
    void col2im_3d_cpu(const Dtype* data_col,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_im, const bool round_down = true);

    template <typename Dtype>
    void col2im_3d_gpu(const Dtype* data_col,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_im, const bool round_down = true);

/**
 * A helper function to calculate the output dimension's size give the original
 * size of the image, padding, patch size and stride.
 * @param  image_size The size of the dimension in the original image
 * @param  pad_size   The amount of padding to apply to the original image
 * @param  patch_size The size of the dimension in the patch taken from the image
 * @param  stride     The patch's stride over the original image
 * @param  round_down Whether to round down or up when calculating the size
 * @return            The output size of the patch image
 * @remarks round_down can be used to control pooling/conv style im2col method.
 */
    inline int dimension_out_size(const int image_size, const int pad_size, const int patch_size,
                                  const int stride, const bool round_down) {
        if (round_down) {
            return (image_size + 2 * pad_size - patch_size) / stride + 1;
        } else {
            return static_cast<int>(std::ceil(static_cast<float>(image_size + 2 * pad_size - patch_size) / stride)) + 1;
        }
    }

    inline
    tensorflow::Status GetSimnetsOutputSizeFromDims(
            tensorflow::shape_inference::InferenceContext* c,
            tensorflow::shape_inference::DimensionHandle input_size,
            tensorflow::shape_inference::DimensionOrConstant filter_size, tensorflow::int64 stride,
            int padding, bool round_down, tensorflow::shape_inference::DimensionHandle* output_size) {
        if (stride <= 0) {
            return tensorflow::errors::InvalidArgument("Stride must be > 0, but got ", stride);
        }

        TF_RETURN_IF_ERROR(c->Add(input_size, 2 * padding, output_size));
        TF_RETURN_IF_ERROR(c->Subtract(*output_size, filter_size, output_size));
        if (!round_down)
            TF_RETURN_IF_ERROR(c->Add(*output_size, stride - 1, output_size));
        TF_RETURN_IF_ERROR(c->Divide(*output_size, stride, false, output_size));
        TF_RETURN_IF_ERROR(c->Add(*output_size, 1, output_size));

        return tensorflow::Status::OK();
    }
}  // namespace simnets_tf

#endif //SIMNETS_TF_IM2COL_HPP
