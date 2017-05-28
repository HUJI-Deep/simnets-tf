#include <vector>
#include <cstring> // for memset

#include "utils/im2col.hpp"

namespace simnets_tf {

    template<typename Dtype>
    void simnets_tf_set(const int N, const Dtype alpha, Dtype *Y) {
        if (alpha == 0) {
            memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
            for (int i = 0; i < N; ++i) {
                Y[i] = alpha;
            }
        }
    }

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
    inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    template <typename Dtype>
    void im2col_cpu(const Dtype* data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    Dtype* data_col) {
        const int output_h = (height + 2 * pad_h -
                              (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w -
                              (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            for (int output_cols = output_w; output_cols; output_cols--) {
                                *(data_col++) = 0;
                            }
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    *(data_col++) = data_im[input_row * width + input_col];
                                } else {
                                    *(data_col++) = 0;
                                }
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
    }

// Explicit instantiation
    template void im2col_cpu<float>(const float* data_im, const int channels,
                                    const int height, const int width, const int kernel_h, const int kernel_w,
                                    const int pad_h, const int pad_w, const int stride_h,
                                    const int stride_w, const int dilation_h, const int dilation_w,
                                    float* data_col);
    template void im2col_cpu<double>(const double* data_im, const int channels,
                                     const int height, const int width, const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w, const int stride_h,
                                     const int stride_w, const int dilation_h, const int dilation_w,
                                     double* data_col);

    template <typename Dtype>
    inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
                                   const int num_spatial_axes, const int* im_shape, const int* col_shape,
                                   const int* kernel_shape, const int* pad, const int* stride,
                                   const int* dilation, Dtype* data_output) {
        if (!im2col) {
            int im_size = im_shape[0];
            for (int i = 0; i < num_spatial_axes; ++i) {
                im_size *= im_shape[1 + i];
            }
            simnets_tf_set(im_size, Dtype(0), data_output);
        }
        int kernel_size = 1;
        for (int i = 0; i < num_spatial_axes; ++i) {
            kernel_size *= kernel_shape[i];
        }
        const int channels_col = col_shape[0];
        std::vector<int> d_offset(num_spatial_axes, 0);
        std::vector<int> d_iter(num_spatial_axes, 0);
        for (int c_col = 0; c_col < channels_col; ++c_col) {
            // Loop over spatial axes in reverse order to compute a per-axis offset.
            int offset = c_col;
            for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
                if (d_i < num_spatial_axes - 1) {
                    offset /= kernel_shape[d_i + 1];
                }
                d_offset[d_i] = offset % kernel_shape[d_i];
            }
            for (bool incremented = true; incremented; ) {
                // Loop over spatial axes in forward order to compute the indices in the
                // image and column, and whether the index lies in the padding.
                int index_col = c_col;
                int index_im = c_col / kernel_size;
                bool is_padding = false;
                for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
                    const int d = d_iter[d_i];
                    const int d_im = d * stride[d_i] - pad[d_i] +
                                     d_offset[d_i] * dilation[d_i];
                    is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
                    index_col *= col_shape[d_i + 1];
                    index_col += d;
                    index_im *= im_shape[d_i + 1];
                    index_im += d_im;
                }
                if (im2col) {
                    if (is_padding) {
                        data_output[index_col] = 0;
                    } else {
                        data_output[index_col] = data_input[index_im];
                    }
                } else if (!is_padding) {  // col2im
                    data_output[index_im] += data_input[index_col];
                }
                // Loop over spatial axes in reverse order to choose an index,
                // like counting.
                incremented = false;
                for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
                    const int d_max = col_shape[d_i + 1];
                    //DCHECK_LT(d_iter[d_i], d_max);
                    if (d_iter[d_i] == d_max - 1) {
                        d_iter[d_i] = 0;
                    } else {  // d_iter[d_i] < d_max - 1
                        ++d_iter[d_i];
                        incremented = true;
                        break;
                    }
                }
            }  // while(incremented) {
        }  // for (int c = 0; c < channels_col; ++c) {
    }

    template <typename Dtype>
    void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
                       const int* im_shape, const int* col_shape,
                       const int* kernel_shape, const int* pad, const int* stride,
                       const int* dilation, Dtype* data_col) {
        const bool kIm2Col = true;
        im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                           kernel_shape, pad, stride, dilation, data_col);
    }

// Explicit instantiation
    template void im2col_nd_cpu<float>(const float* data_im,
                                       const int num_spatial_axes,
                                       const int* im_shape, const int* col_shape,
                                       const int* kernel_shape, const int* pad, const int* stride,
                                       const int* dilation, float* data_col);
    template void im2col_nd_cpu<double>(const double* data_im,
                                        const int num_spatial_axes,
                                        const int* im_shape, const int* col_shape,
                                        const int* kernel_shape, const int* pad, const int* stride,
                                        const int* dilation, double* data_col);

    template <typename Dtype>
    void col2im_cpu(const Dtype* data_col, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w,
                    const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    Dtype* data_im) {
        simnets_tf_set(height * width * channels, Dtype(0), data_im);
        const int output_h = (height + 2 * pad_h -
                              (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w -
                              (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
        const int channel_size = height * width;
        for (int channel = channels; channel--; data_im += channel_size) {
            for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                    int input_row = -pad_h + kernel_row * dilation_h;
                    for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                            data_col += output_w;
                        } else {
                            int input_col = -pad_w + kernel_col * dilation_w;
                            for (int output_col = output_w; output_col; output_col--) {
                                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                    data_im[input_row * width + input_col] += *data_col;
                                }
                                data_col++;
                                input_col += stride_w;
                            }
                        }
                        input_row += stride_h;
                    }
                }
            }
        }
    }

// Explicit instantiation
    template void col2im_cpu<float>(const float* data_col, const int channels,
                                    const int height, const int width, const int kernel_h, const int kernel_w,
                                    const int pad_h, const int pad_w, const int stride_h,
                                    const int stride_w, const int dilation_h, const int dilation_w,
                                    float* data_im);
    template void col2im_cpu<double>(const double* data_col, const int channels,
                                     const int height, const int width, const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w, const int stride_h,
                                     const int stride_w, const int dilation_h, const int dilation_w,
                                     double* data_im);

    template <typename Dtype>
    void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
                       const int* im_shape, const int* col_shape,
                       const int* kernel_shape, const int* pad, const int* stride,
                       const int* dilation, Dtype* data_im) {
        const bool kIm2Col = false;
        im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                           kernel_shape, pad, stride, dilation, data_im);
    }

// Explicit instantiation
    template void col2im_nd_cpu<float>(const float* data_col,
                                       const int num_spatial_axes,
                                       const int* im_shape, const int* col_shape,
                                       const int* kernel_shape, const int* pad, const int* stride,
                                       const int* dilation, float* data_im);
    template void col2im_nd_cpu<double>(const double* data_col,
                                        const int num_spatial_axes,
                                        const int* im_shape, const int* col_shape,
                                        const int* kernel_shape, const int* pad, const int* stride,
                                        const int* dilation, double* data_im);


// 3D Convolutions


    template <typename Dtype>
    void im2col_3d_cpu(const Dtype* data_im,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_col,
                       const bool round_down, const Dtype out_of_bounds_value) {
        int height_col = dimension_out_size(height, pad_h, patch_h, stride_h, round_down);
        int width_col = dimension_out_size(width, pad_w, patch_w, stride_w, round_down);
        int channels_col = dimension_out_size(channels, pad_c, patch_c, stride_c, round_down);
        int patch_col = patch_c * patch_h * patch_w;
        for (int p = 0; p < patch_col; ++p) {
            int w_offset = p % patch_w;
            int h_offset = (p / patch_w) % patch_h;
            int c_offset = p / patch_h / patch_w;
            for (int c = 0; c < channels_col; ++c) {
                for (int h = 0; h < height_col; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        int h_pad = h * stride_h - pad_h + h_offset;
                        int w_pad = w * stride_w - pad_w + w_offset;
                        int c_pad = c * stride_c - pad_c + c_offset;
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
                            && c_pad >= 0 && c_pad < channels) {
                            data_col[((p * channels_col + c) * height_col + h) * width_col + w] =
                                    data_im[(c_pad * height + h_pad) * width + w_pad];
                        } else {
                            data_col[((p * channels_col + c) * height_col + h) * width_col + w] = out_of_bounds_value;
                        }
                    }
                }
            }
        }
    }

// Explicit instantiation
    template void im2col_3d_cpu(const float* data_im,
                                const int channels, const int height, const int width,
                                const int patch_c, const int patch_h, const int patch_w,
                                const int pad_c, const int pad_h, const int pad_w,
                                const int stride_c, const int stride_h, const int stride_w,
                                float* data_col,
                                const bool round_down, const float out_of_bounds_value);
    template void im2col_3d_cpu(const double* data_im,
                                const int channels, const int height, const int width,
                                const int patch_c, const int patch_h, const int patch_w,
                                const int pad_c, const int pad_h, const int pad_w,
                                const int stride_c, const int stride_h, const int stride_w,
                                double* data_col,
                                const bool round_down, const double out_of_bounds_value);
    template <typename Dtype>
    void col2im_3d_cpu(const Dtype* data_col,
                       const int channels, const int height, const int width,
                       const int patch_c, const int patch_h, const int patch_w,
                       const int pad_c, const int pad_h, const int pad_w,
                       const int stride_c, const int stride_h, const int stride_w,
                       Dtype* data_im, const bool round_down) {
        simnets_tf_set(height * width * channels, Dtype(0), data_im);
        int height_col = dimension_out_size(height, pad_h, patch_h, stride_h, round_down);
        int width_col = dimension_out_size(width, pad_w, patch_w, stride_w, round_down);
        int channels_col = dimension_out_size(channels, pad_c, patch_c, stride_c, round_down);
        int patch_col = patch_c * patch_h * patch_w;
        for (int p = 0; p < patch_col; ++p) {
            int w_offset = p % patch_w;
            int h_offset = (p / patch_w) % patch_h;
            int c_offset = p / patch_h / patch_w;
            for (int c = 0; c < channels_col; ++c) {
                for (int h = 0; h < height_col; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        int h_pad = h * stride_h - pad_h + h_offset;
                        int w_pad = w * stride_w - pad_w + w_offset;
                        int c_pad = c * stride_c - pad_c + c_offset;
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
                            && c_pad >= 0 && c_pad < channels) {
                            data_im[(c_pad * height + h_pad) * width + w_pad] +=
                                    data_col[((p * channels_col + c) * height_col + h) * width_col + w];
                        }
                    }
                }
            }
        }
    }

    template void col2im_3d_cpu(const float* data_col,
                                const int channels, const int height, const int width,
                                const int patch_c, const int patch_h, const int patch_w,
                                const int pad_c, const int pad_h, const int pad_w,
                                const int stride_c, const int stride_h, const int stride_w,
                                float* data_im, const bool round_down);
    template void col2im_3d_cpu(const double* data_col,
                                const int channels, const int height, const int width,
                                const int patch_c, const int patch_h, const int patch_w,
                                const int pad_c, const int pad_h, const int pad_w,
                                const int stride_c, const int stride_h, const int stride_w,
                                double* data_im, const bool round_down);

}  // namespace simnets_tf