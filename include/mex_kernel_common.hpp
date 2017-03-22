#ifndef SIMNETS_TF_MEX_LAYER_COMMON_HPP
#define SIMNETS_TF_MEX_LAYER_COMMON_HPP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "im2col.hpp"


class MEXKernelCommon {
public:
    MEXKernelCommon(tensorflow::OpKernelConstruction* context);

private:
    void CalculateDimensions(tensorflow::OpKernelContext* context);
    int block_c_, block_h_, block_w_;
    int stride_c_, stride_h_, stride_w_;
    int num_;
    int pad_c_, pad_h_, pad_w_;
    int channels_, height_, width_;
    int num_instances_;
    int channels_out_, height_out_, width_out_;
    float blocks_out_of_bounds_value_;
    bool blocks_round_down_;
    bool use_log_space_parameters_;
    float linear_space_min_value_;
    int shared_offsets_region_c_, shared_offsets_region_h_, shared_offsets_region_w_;
    bool use_unshared_regions_;
    int unshared_offsets_region_c_, unshared_offsets_region_h_, unshared_offsets_region_w_;
    int offsets_c_, offsets_h_, offsets_w_;
    int region_size_;
    int num_regions_;
    bool is_1x1_;
    bool is_2D_pooling_;
    bool normalize_patches_;
    float normalization_fudge_factor_;
    bool normalize_variance_;
    bool normalize_offsets_;
    bool normalize_offsets_projected_;
    bool softmax_mode_;
    int exp_iterations_, log_iterations_;
    float exp_approx_param_, log_approx_param_;
    MEXParameter_ExpApproximation exp_method_;
    MEXParameter_LogApproximation log_method_;
    bool param_initialized_;
    bool expects_labels_;
    float maximum_entropy_regularization_coeff_;

    /// M_ is the channel dimension of the output for a single group, which is the
    /// leading dimension of the filter matrix.
    int M_;
    /// K_ is the dimension of an unrolled input for a single group, which is the
    /// leading dimension of the data matrix.
    int K_;
    /// N_ is the spatial dimension of the output, the H x W, which are the last
    /// dimensions of the data and filter matrices.
    int N_;

    static MEXParams Calculate(tensorflow::OpKernelContext *context);
};
/**
 * @brief MEX Layer as defined in the SimNets article. Generelizes pooling, relu and labeling.
 */
template <typename Dtype>
class MEXLayer : public Layer<Dtype> {
public:
    /**
     * @param param provides SimilarityParameter similarity_param,
     *    with BlockParameter options:
     *  - num_instances. The number of templates/weights.
     *  - block_size / block_h / block_w / block_c. The filter dimensions, given by
     *  block_size for square filters or block_h and block_w and block_c for general
     *  block shapes.
     *  - stride / stride_h / stride_w (\b optional, default 1) / stride_c
     *  (\b optional, default match block_c). The filter stride, given by stride_size
     *  for equal dimensions or stride_h and stride_w for different strides.
     *  By default the blocks are dense with stride 1.
     *  - pad / pad_h / pad_w / pad_c (\b optional, default 0). The zero-padding for
     *  convolution, given by pad for equal dimensions or pad_h and pad_w for
     *  different padding. Input padding is computed implicitly instead of
     *  actually padding.
     *  - group (\b optional, default 1). The number of template groups. Group
     *  similarity is a method for reducing parameterization by selectively
     *  connecting input and output channels. The input and output channel dimensions must be divisible
     *  by the number of groups. For group @f$ \geq 1 @f$, the
     *  similarity filters' input and output channels are separated s.t. each
     *  group takes 1 / group of the input channels and makes 1 / group of the
     *  output channels. Concretely 4 input channels, 8 output channels, and
     *  2 groups separate input channels 1-2 and output channels 1-4 into the
     *  first group and input channels 3-4 and output channels 5-8 into the second
     *  group.
     *  - bias_term (\b optional, default true). Whether to have a bias.
     */
    explicit MEXLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const { return "MEX"; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline bool EqualNumBottomTopBlobs() const {
        return this->layer_param_.mex_param().unsupervised_init().type() != "labeling_density";
    }
    virtual bool needs_unsupervised_init();
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual Dtype test_init_step_objective_cpu(const vector<Blob<Dtype>*>& bottom);
    virtual Dtype test_init_step_objective_gpu(const vector<Blob<Dtype>*>& bottom);
    virtual bool init_step_cpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective);
    virtual bool init_step_gpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective);


    Blob<Dtype> col_buffer_;
    Blob<Dtype> row_buffer_;
    Blob<Dtype> normed_offsets_;
    shared_ptr<SyncedMemory> transposed_offsets_;
    Blob<Dtype> offsets_norm_factor_;
    shared_ptr<SyncedMemory> epsilon_helper_;
    Blob<Dtype> one_zero_vec_;

    Blob<Dtype> split_patches_in_;
    Blob<Dtype> split_patches_out_;
    shared_ptr<SyncedMemory> split_patches_out_inter_;
    vector<shared_ptr<Blob<Dtype> > > input_for_learner_;
    shared_ptr<UnsupervisedLearner<Dtype> > unsupervised_learner_;
};


#endif  // SIMNETS_TF_MEX_LAYER_COMMON_HPP