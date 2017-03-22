#include "mex_kernel_common.hpp"
#include "im2col.hpp"

using namespace tensorflow;

MEXKernelCommon::MEXKernelCommon(tensorflow::OpKernelConstruction *context)
        : tensorflow::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("softmax_mode", &softmax_mode_));

    OP_REQUIRES_OK(context, context->GetAttr("num_instances", &num_instances_));
    OP_REQUIRES(context, num_instances_ > 0, errors::InvalidArgument("num_instances must be positive"));

    // TODO: How to handle pad dimensions?
    std::vector<int> padding;
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    if (padding.size() == 1) {
        pad_h_ = pad_w_ = pad_c_ = padding[0];
    } else {
        pad_h_ = block_param.pad_h();
        pad_w_ = block_param.pad_w();
    }
    pad_c_ = block_param.pad_c();

    OP_REQUIRES_OK(context, context->GetAttr("blocks_out_of_bounds_value", &blocks_out_of_bounds_value_));
    OP_REQUIRES_OK(context, context->GetAttr("blocks_round_down", &blocks_round_down_));

    //OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    //std::string similarity_func_str;
    //OP_REQUIRES_OK(context, context->GetAttr("similarity_function", &similarity_func_str));
    //if (similarity_func_str == "L1") {
    //    similarity_function_ = SIM_FUNC_L1;
    //} else if (similarity_func_str == "L2") {
    //    similarity_function_ = SIM_FUNC_L2;
    //} else {
    //    assert(false); // Bad similarity function
    //}

    //OP_REQUIRES_OK(context, context->GetAttr("normalization_term", &normalization_term_));
    //OP_REQUIRES_OK(context, context->GetAttr("normalization_term_fudge", &normalization_term_fudge_));
    //OP_REQUIRES_OK(context, context->GetAttr("ignore_nan_input", &ignore_nan_input_));
    //OP_REQUIRES_OK(context, context->GetAttr("out_of_bounds_value", &out_of_bounds_value_));
}

void MEXKernelCommon::CalculateDimensions(OpKernelContext *context) {

    // Configure input size and number of instances
    channels_ = context->input(0).shape()[1];
    height_ = context->input(0).shape()[2];
    width_ = context->input(0).shape()[3];

    // TODO: Where can i get the block dimensions?
    block_h_ = context->input(1).shape()[1];
    block_w_ = block_param.block_size();
    block_c_ = block_param.block_c();

    //CHECK_GT(block_c_, 0) << "Filter dimensions cannot be zero.";
    //CHECK_LE(block_c_, channels_)
    //<< "Filter dimensions cannot be exceed channel dimention";
    //CHECK_GT(block_h_, 0) << "Filter dimensions cannot be zero.";
    //CHECK_GT(block_w_, 0) << "Filter dimensions cannot be zero.";

    if (!block_param.

    has_stride_h()

    ) {
    stride_h_ = stride_w_ = block_param.stride();
    } else {
    stride_h_ = block_param.stride_h();
    stride_w_ = block_param.stride_w();
    }
    stride_c_ = block_param.stride_c();
    if (stride_c_ < 0) {
    stride_c_ = block_c_;
    }

    height_out_ = simnets_tf::dimension_out_size(height_, pad_h_, block_h_, stride_h_, blocks_round_down_);
    width_out_ = simnets_tf::dimension_out_size(width_, pad_w_, block_w_, stride_w_, blocks_round_down_);
    channels_out_ = simnets_tf::dimension_out_size(channels_, pad_c_, block_c_, stride_c_, blocks_round_down_);

    // Handle the parameters: templatess and biases.
    // - blobs_[0] holds the MEX's parameter
    // - blobs_[1] holds the MEX's offsets

    // TODO: Delete them, right?
    normalize_offsets_ = mex_param.normalize_offsets();
    normalize_offsets_projected_ = mex_param.normalize_offsets_projected();
    use_log_space_parameters_ = mex_param.use_log_space_parameters();
    linear_space_min_value_ = mex_param.linear_space_min_value();

    if (!use_log_space_parameters_) {
    CHECK_GT(linear_space_min_value_, 0)
    << "The minimum value of the liear space must be positive.";
    }

    use_unshared_regions_ = mex_param.use_unshared_regions();
    if (!use_unshared_regions_) {
    CHECK(!(mex_param.has_shared_offsets_region_size()
            && mex_param.has_shared_offsets_region_h()
            && mex_param.has_shared_offsets_region_w()))
    << "Shared offsets size is shared_offsets_region_size OR shared_offsets_region_h and shared_offsets_region_w; not both";
    CHECK((!(mex_param.has_shared_offsets_region_h()) && !(mex_param.has_shared_offsets_region_w())) ||
          (mex_param.has_shared_offsets_region_h() && mex_param.has_shared_offsets_region_w()))
    << "For non-square shared offsets, both shared_offsets_region_h and shared_offsets_region_w are required.";
    if (mex_param.has_shared_offsets_region_h() && mex_param.has_shared_offsets_region_w()) {
        shared_offsets_region_h_ = mex_param.shared_offsets_region_h();
        shared_offsets_region_w_ = mex_param.shared_offsets_region_w();
    } else {
        shared_offsets_region_w_ = shared_offsets_region_h_ = mex_param.shared_offsets_region_size();
    }
    shared_offsets_region_c_ = mex_param.shared_offsets_region_c();
    if (shared_offsets_region_h_ < 0) {
        shared_offsets_region_h_ = height_out_;
    }
    shared_offsets_region_h_ = std::min(shared_offsets_region_h_, height_out_);
    if (shared_offsets_region_w_ < 0) {
    shared_offsets_region_w_ = width_out_;
    }
    shared_offsets_region_w_ = std::min(shared_offsets_region_w_, width_out_);
    if (shared_offsets_region_c_ < 0) {
    shared_offsets_region_c_ = channels_out_;
    }
    shared_offsets_region_c_ = std::min(shared_offsets_region_c_, channels_out_);
    CHECK_GT(shared_offsets_region_h_, 0);
    CHECK_GT(shared_offsets_region_w_, 0);
    CHECK_GT(shared_offsets_region_c_, 0);
    offsets_h_ = caffe_ceiled_div(height_out_, shared_offsets_region_h_);
    offsets_w_ = caffe_ceiled_div(width_out_, shared_offsets_region_w_);
    offsets_c_ = caffe_ceiled_div(channels_out_, shared_offsets_region_c_);
    } else {
    CHECK(!(mex_param.has_unshared_offsets_region_size()
            && mex_param.has_unshared_offsets_region_h()
            && mex_param.has_unshared_offsets_region_w()))
    << "Unshared offsets size is shared_offsets_region_size OR shared_offsets_region_h and shared_offsets_region_w; not both";
    CHECK((!(mex_param.has_unshared_offsets_region_h()) && !(mex_param.has_unshared_offsets_region_w())) ||
          (mex_param.has_unshared_offsets_region_h() && mex_param.has_unshared_offsets_region_w()))
    << "For non-square unshared offsets, both unshared_offsets_region_h and unshared_offsets_region_w are required.";
    if (mex_param.

    has_unshared_offsets_region_h()

    && mex_param.

    has_unshared_offsets_region_w()

    ) {
    unshared_offsets_region_h_ = mex_param.unshared_offsets_region_h();
    unshared_offsets_region_w_ = mex_param.unshared_offsets_region_w();
    } else {
    unshared_offsets_region_w_ = unshared_offsets_region_h_ = mex_param.unshared_offsets_region_size();
    }
    unshared_offsets_region_c_ = mex_param.unshared_offsets_region_c();
    if (unshared_offsets_region_h_ < 0) {
    unshared_offsets_region_h_ = height_out_;
    }
    unshared_offsets_region_h_ = std::min(unshared_offsets_region_h_, height_out_);
    if (unshared_offsets_region_w_ < 0) {
    unshared_offsets_region_w_ = width_out_;
    }
    unshared_offsets_region_w_ = std::min(unshared_offsets_region_w_, width_out_);
    if (unshared_offsets_region_c_ < 0) {
    unshared_offsets_region_c_ = channels_out_;
    }
    unshared_offsets_region_c_ = std::min(unshared_offsets_region_c_, channels_out_);
    CHECK_GT(unshared_offsets_region_h_, 0);
    CHECK_GT(unshared_offsets_region_w_, 0);
    CHECK_GT(unshared_offsets_region_c_, 0);
    offsets_h_ = unshared_offsets_region_h_;
    offsets_w_ = unshared_offsets_region_w_;
    offsets_c_ = unshared_offsets_region_c_;
    shared_offsets_region_h_ = caffe_ceiled_div(height_out_, unshared_offsets_region_h_);
    shared_offsets_region_w_ = caffe_ceiled_div(width_out_, unshared_offsets_region_w_);
    shared_offsets_region_c_ = caffe_ceiled_div(channels_out_, unshared_offsets_region_c_);
    }
    num_regions_ = offsets_h_ * offsets_w_ * offsets_c_;
    region_size_ = shared_offsets_region_w_ * shared_offsets_region_h_ * shared_offsets_region_c_;
    // Prepare the matrix multiplication computation.
    // Each input will be convolved as a single GEMM.
    M_ = num_instances_;
    K_ = block_c_ * block_h_ * block_w_;
    N_ = height_out_ * width_out_ * channels_out_;
}
//param_initialized_ = false;
//if (this->blobs_.size() > 0) {
//LOG(INFO) << "Skipping parameter initialization";
//param_initialized_ = true;
//Blob<Dtype> temp(this->blobs_[1]->num(),
//                 this->blobs_[1]->channels(),
//                 this->blobs_[1]->height(),
//                 this->blobs_[1]->width());
//caffe_copy<Dtype>(temp.count(), this->blobs_[1]->cpu_data(), temp.mutable_cpu_data());
//this->blobs_[1]->SetPadding(ggemm_padded_output_size(M_, K_));
//caffe_copy<Dtype>(temp.count(), temp.cpu_data(), this->blobs_[1]->mutable_cpu_data());
//} else {
//this->blobs_.resize(2);
//// Initialize the epsilon parameter
//this->blobs_[0].reset(new Blob<Dtype>(
//1, 1, 1, 1));
//shared_ptr<Filler<Dtype> > epsilon_filler(GetFiller<Dtype>(
//        this->layer_param_.mex_param().epsilon_filler()));
//epsilon_filler->Fill(this->blobs_[0].get());
//// Initialize and fill the offsets:
//// every offset matrix (one for each shared region) is a marix num_instance x K
//// the matrices are arranged on top of one another
//this->blobs_[1].reset(new Blob<Dtype>(num_regions_, 1, num_instances_, K_));
//this->blobs_[1]->SetPadding(ggemm_padded_output_size(M_, K_));
//UnsupervisedInitialization init_param = mex_param.unsupervised_init();
//const std::string& type = init_param.type();
//if (type == "labeling_density") {
//expects_labels_ = true;
//CHECK_EQ(bottom.size(), top.size() * 2);
//unsupervised_learner_.reset(new LabelingDensityLearner<Dtype>(
//        num_instances_, init_param.num_batches(), init_param.max_iterations(), init_param.fudge_factor(),
//        init_param.soft_assignment(), init_param.labeling_density_lambda()));
//input_for_learner_.resize(2);
//input_for_learner_[0].reset(new Blob<Dtype>(1,1,1,1));
//input_for_learner_[1].reset(new Blob<Dtype>(1,1,1,1));
//} else {
//shared_ptr<Filler<Dtype> > offsets_filler(GetFiller<Dtype>(
//        mex_param.offsets_filler()));
//offsets_filler->Fill(this->blobs_[1].get());
//param_initialized_ = true;
//}
//}
//if (normalize_offsets_) {
//normed_offsets_.ReshapeLike(*this->blobs_[1]);
//normed_offsets_.SetPadding(this->blobs_[1]->padding());
//offsets_norm_factor_.Reshape(num_regions_, 1, num_instances_, 1);
//offsets_norm_factor_.SetPadding(ggemm_padded_output_size(num_instances_ * num_regions_, 1));
//}
//transposed_offsets_.reset(new SyncedMemory((this->blobs_[1]->count() + this->blobs_[1]->padding()) * sizeof(Dtype)));
//epsilon_helper_.reset(new SyncedMemory((num_regions_ * M_ * K_ + ggemm_padded_output_size(M_, K_)) * sizeof(Dtype)));
//// Propagate gradients to the parameters (as directed by backward pass).
//this->param_propagate_down_.resize(this->blobs_.size(), true);
//
//// Special case: im2col is the identity for 1x1xchannels convolution
//// with stride 1 in 2D and no padding, so flag for skipping the buffer
//// and transformation.
//is_1x1_ = (block_c_ == channels_ || (block_c_ == 1 && stride_c_ == 1)) && block_w_ == 1 && block_h_ == 1
//          && stride_h_ == 1 && stride_w_ == 1
//          && pad_c_ == 0 && pad_h_ == 0 && pad_w_ == 0;
//is_2D_pooling_ = num_instances_ == 1
//                 && block_c_ == 1 && stride_c_ == 1 && pad_c_ == 0
//                 && !normalize_patches_ && !normalize_offsets_;
//if (normalize_patches_) {
//row_buffer_.Reshape(block_c_ * block_h_ * block_w_,
//        channels_out_, height_out_, width_out_);
//}
//}
//
//template <typename Dtype>
//void MEXLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//const vector<Blob<Dtype>*>& top) {
//// TODO: generalize to handle inputs of different shapes.
//num_ = bottom[0]->num();
//for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
//CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
//if (expects_labels_ && bottom_id % 2 == 1) continue;
//CHECK_EQ(channels_, bottom[bottom_id]->channels())
//<< "Inputs must have same channels.";
//CHECK_EQ(height_, bottom[bottom_id]->height())
//<< "Inputs must have same height.";
//CHECK_EQ(width_, bottom[bottom_id]->width())
//<< "Inputs must have same width.";
//}
//// Shape the tops.
//for (int top_id = 0; top_id < top.size(); ++top_id) {
//top[top_id]->Reshape(
//        num_, num_instances_*channels_out_, height_out_, width_out_);
//top[top_id]->SetPadding(ggemm_padded_output_size(M_, region_size_));
//}
//// The im2col result buffer will only hold one image at a time to avoid
//// overly large memory usage.
//col_buffer_.Reshape(K_, channels_out_, height_out_, width_out_);
//col_buffer_.SetPadding(ggemm_padded_output_size(K_, N_));
//int sum_size = std::max(M_,
//                        std::max(K_,
//                                 region_size_));
//if (!param_initialized_ && this->layer_param_.mex_param().unsupervised_init().type() == "labeling_density") {
//sum_size = std::max(sum_size, N_);
//}
//one_zero_vec_.Reshape(1, 1, 1, sum_size);
//caffe_set(sum_size, Dtype(1), one_zero_vec_.mutable_cpu_data());
//caffe_set(sum_size, Dtype(0), one_zero_vec_.mutable_cpu_diff());
//
//if (num_regions_ > 1) {
//split_patches_in_.Reshape(num_regions_, 1, K_, region_size_);
//split_patches_in_.SetPadding(ggemm_padded_output_size(K_, region_size_));
//split_patches_out_.Reshape(num_regions_, 1, M_, region_size_);
//split_patches_out_.SetPadding(ggemm_padded_output_size(M_, region_size_));
//}
//split_patches_out_inter_.reset(new SyncedMemory(num_regions_ * M_ * region_size_ * sizeof(typename vec<Dtype>::vec2)));
//}
//
//
//template <typename Dtype>
//bool MEXLayer<Dtype>::needs_unsupervised_init() {
//    if (param_initialized_) {
//        return false;
//    }
//    MEXParameter mex_param = this->layer_param_.mex_param();
//    if (!mex_param.has_unsupervised_init()) {
//        return false;
//    }
//    UnsupervisedInitialization init_param = mex_param.unsupervised_init();
//    const std::string& type = init_param.type();
//    if (type == "none") {
//        return false;
//    } else {
//        return true;
//    }
//}
//
//template <typename Dtype>
//Dtype MEXLayer<Dtype>::test_init_step_objective_cpu(const vector<Blob<Dtype>*>& bottom) {
//if (!needs_unsupervised_init()) {
//return INFINITY;
//}
//
//int batch_size = 0;
//for (int i = 0; i < bottom.size(); ++i) {
//if (expects_labels_ && i % 2 == 1) continue;
//batch_size += N_ * bottom[i]->num();
//}
//
//input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
//if (expects_labels_) {
//input_for_learner_[1]->Reshape(batch_size, 1, 1, 1);
//}
//
//Dtype* patches_data = input_for_learner_[0]->mutable_cpu_data();
//for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
//if (expects_labels_ && bottom_idx % 2 == 1) continue;
//const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
//Dtype* col_buff = NULL;
//if (!is_1x1_ || normalize_patches_) {
//col_buff = col_buffer_.mutable_cpu_data();
//}
//for (int n = 0; n < num_; ++n) {
//// im2col transformation: unroll input regions for filtering
//// into column matrix for multplication.
//if (!is_1x1_) {
//im2col_3d_cpu(
//        bottom_data + bottom[bottom_idx]->offset(n),
//        channels_, height_, width_,
//        block_c_, block_h_, block_w_,
//        pad_c_, pad_h_, pad_w_,
//        stride_c_, stride_h_, stride_w_,
//        col_buff,
//        blocks_round_down_, blocks_out_of_bounds_value_);
//} else {  // special case for 1x1 convolution
//if (!normalize_patches_) {
//col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
//} else {
//caffe_copy(N_ * K_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
//}
//}
//
//if (normalize_patches_) {
//caffe_cpu_transpose(K_, N_,
//        col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
//caffe_cpu_normalize_patches_rows_forward(K_, N_,
//        normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
//} else {
//caffe_cpu_transpose(K_, N_,
//        col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
//}
//}
//}
//if (expects_labels_) {
//Dtype* labels_data = input_for_learner_[1]->mutable_cpu_data();
//for (int bottom_idx = 1; bottom_idx < bottom.size(); bottom_idx += 2) {
//const Dtype* labels = bottom[bottom_idx]->cpu_data();
//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, N_, 1,
//Dtype(1), labels, one_zero_vec_.cpu_data(),
//        Dtype(0), labels_data + ((bottom_idx - 1) / 2) * num_ * N_);
//}
//}
//return unsupervised_learner_->objective_cpu(input_for_learner_);
//}
//
//template <typename Dtype>
//bool MEXLayer<Dtype>::init_step_cpu(const vector<Blob<Dtype>*>& bottom, Dtype* objective) {
//if (!needs_unsupervised_init()) {
//return false;
//}
//int batch_size = 0;
//for (int i = 0; i < bottom.size(); ++i) {
//if (expects_labels_ && i % 2 == 1) continue;
//batch_size += N_ * bottom[i]->num();
//}
//
//input_for_learner_[0]->Reshape(batch_size, K_, 1, 1);
//if (expects_labels_) {
//input_for_learner_[1]->Reshape(batch_size, 1, 1, 1);
//}
//
//Dtype* patches_data = input_for_learner_[0]->mutable_cpu_data();
//for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
//if (expects_labels_ && bottom_idx % 2 == 1) continue;
//const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
//Dtype* col_buff = NULL;
//if (!is_1x1_ || normalize_patches_) {
//col_buff = col_buffer_.mutable_cpu_data();
//}
//for (int n = 0; n < num_; ++n) {
//// im2col transformation: unroll input regions for filtering
//// into column matrix for multplication.
//if (!is_1x1_) {
//im2col_3d_cpu(
//        bottom_data + bottom[bottom_idx]->offset(n),
//        channels_, height_, width_,
//        block_c_, block_h_, block_w_,
//        pad_c_, pad_h_, pad_w_,
//        stride_c_, stride_h_, stride_w_,
//        col_buff,
//        blocks_round_down_, blocks_out_of_bounds_value_);
//} else {  // special case for 1x1 convolution
//if (!normalize_patches_) {
//col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
//} else {
//caffe_copy(N_ * K_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
//}
//}
//if (normalize_patches_) {
//caffe_cpu_transpose(K_, N_,
//        col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
//caffe_cpu_normalize_patches_rows_forward(K_, N_,
//        normalization_fudge_factor_, patches_data + (bottom_idx * num_ + n) * K_ * N_, normalize_variance_);
//} else {
//caffe_cpu_transpose(K_, N_,
//        col_buff, patches_data + (bottom_idx * num_ + n) * K_ * N_);
//}
//}
//}
//if (expects_labels_) {
//Dtype* labels_data = input_for_learner_[1]->mutable_cpu_data();
//for (int bottom_idx = 1; bottom_idx < bottom.size(); bottom_idx += 2) {
//const Dtype* labels = bottom[bottom_idx]->cpu_data();
//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, N_, 1,
//Dtype(1), labels, one_zero_vec_.cpu_data(),
//        Dtype(0), labels_data + ((bottom_idx - 1) / 2) * num_ * N_);
//}
//}
//bool not_finished = unsupervised_learner_->step_cpu(input_for_learner_, objective);
//if (!not_finished) {
//const vector<shared_ptr<Blob<Dtype> > > blobs(1, this->blobs_[1]);
//unsupervised_learner_->fill_cpu(blobs);
//
//// Release memory use for initialization
//for (int i = 0; i < input_for_learner_.size(); ++i) {
//input_for_learner_[i].reset();
//}
//input_for_learner_.clear();
//unsupervised_learner_.reset();
//
//param_initialized_ = true;
//}
//return not_finished;
//}
//
//template <typename Dtype, bool REVERSE>
//void split_patches_cpu(const int N, const int Dim,
//                       const int W, const int H, const int C,
//                       const int W_Gs, const int H_Gs, const int C_Gs,
//                       const int W_Step, const int H_Step, const int C_Step,
//                       typename std::conditional<REVERSE, Dtype*, const Dtype*>::type in,
//                       Dtype* out, const bool use_unshared_regions_) {
//    const int step_out = C_Step * H_Step * W_Step;
//    const int group_step_w = !use_unshared_regions_ ? W_Step : 1;
//    const int group_step_h = !use_unshared_regions_ ? H_Step : 1;
//    const int group_step_c = !use_unshared_regions_ ? C_Step : 1;
//    const int region_step_w = !use_unshared_regions_ ? 1 : W_Gs;
//    const int region_step_h = !use_unshared_regions_ ? 1 : H_Gs;
//    const int region_step_c = !use_unshared_regions_ ? 1 : C_Gs;
//    Dtype* in_unconst = NULL;
//    if (REVERSE) {
//        in_unconst = (Dtype*)in;
//    }
//    for (int w_g = 0; w_g < W_Gs; ++w_g) {
//        for (int h_g = 0; h_g < H_Gs; ++h_g) {
//            for (int c_g = 0; c_g < C_Gs; ++c_g) {
//                Dtype* o = out + ((c_g * H_Gs + h_g) * W_Gs + w_g) * step_out * Dim;
//                const int group_addr = (c_g * group_step_c * H + h_g * group_step_h) * W + w_g * group_step_w;
//                for (int l = 0; l < C_Step; ++l) {
//                    for (int j = 0; j < H_Step; ++j) {
//                        for (int i = 0; i < W_Step; ++i) {
//                            const int base_addr_out = (l * H_Step + j) * W_Step + i;
//                            const int base_addr_in  = group_addr + (l * region_step_c * H + j * region_step_h) * W  + i * region_step_w;
//                            if (w_g * W_Step + i >= W ||
//                                h_g * H_Step + j >= H ||
//                                c_g * C_Step + l >= C) {
//                                continue;
//                            }
//                            for (int k = 0; k < Dim; ++k) {
//                                if (!REVERSE) {
//                                    o[base_addr_out + k * step_out] = in[base_addr_in + k * N];
//                                } else {
//                                    in_unconst[base_addr_in + k * N] = o[base_addr_out + k * step_out];
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}
//
//template <typename Dtype>
//void mex_forward_cpu(const int M, const int N, const int K, const bool softmax_mode,
//                     const Dtype epsilon, const Dtype* offsets, const Dtype* in, Dtype* out, const int batch_size = 1) {
//    const Dtype init_value = epsilon > 0 ? -INFINITY : INFINITY;
//    if (epsilon > 0) {
//        ggemm_cpu
//        <Dtype, Dtype, Dtype, uint8_t,
//                ggemm_add<Dtype, uint8_t>, ggemm_max<Dtype>, false,
//                true, true, true>
//                            (M, N, K, offsets, in, out,
//                                    init_value, 0, batch_size);
//    } else {
//        ggemm_cpu
//        <Dtype, Dtype, Dtype, uint8_t,
//                ggemm_add<Dtype, uint8_t>, ggemm_min<Dtype>, false,
//                true, true, true>
//                            (M, N, K, offsets, in, out,
//                                    init_value, 0, batch_size);
//    }
//    if (std::isfinite(epsilon)) {
//        ggemm_readc_cpu
//        <false, false, Dtype, Dtype, Dtype, typename vec<Dtype>::vec2,
//                mex_forward_exp<Dtype>, ggemm_add<Dtype>, true, mex_forward_out<Dtype>, true,
//                true, true, true>
//                            (M, N, K, offsets, in, out, out,
//                                    0, make_vec2<Dtype>(epsilon, softmax_mode ? Dtype(0) : (Dtype)-std::log(K)), batch_size);
//    }
//}
//
//template <typename Dtype>
//void MEXLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//const vector<Blob<Dtype>*>& top) {
//Dtype* col_buff = NULL;
//if (!is_1x1_ || normalize_patches_) {
//col_buff = col_buffer_.mutable_cpu_data();
//}
//const Dtype epsilon = this->blobs_[0]->cpu_data()[0];
//Dtype* split_patches_in = NULL;
//Dtype* split_patches_out = NULL;
//const Dtype* offsets = this->blobs_[1]->cpu_data();
//if (!use_log_space_parameters_) {
//caffe_cpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data(), linear_space_min_value_);
//caffe_log<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data());
//}
//if (normalize_offsets_) {
//mex_forward_cpu<Dtype>(M_ * num_regions_, 1, K_, softmax_mode_, epsilon,
//offsets, one_zero_vec_.cpu_diff(), offsets_norm_factor_.mutable_cpu_data());
//Dtype* offsets_mutable = NULL;
//if (!normalize_offsets_projected_) {
//caffe_copy<Dtype>(num_regions_ * M_ * K_, offsets, normed_offsets_.mutable_cpu_data());
//offsets = normed_offsets_.cpu_data();
//offsets_mutable = normed_offsets_.mutable_cpu_data();
//} else {
//offsets_mutable = this->blobs_[1]->mutable_cpu_data();
//}
//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_ * num_regions_, K_, 1,
//-1, offsets_norm_factor_.cpu_data(), one_zero_vec_.cpu_data(),
//1, offsets_mutable);
//}
//
//for (int bottom_idx = 0; bottom_idx < bottom.size(); ++bottom_idx) {
//const int top_idx = expects_labels_ ? bottom_idx / 2 : bottom_idx;
//const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
//Dtype* top_data = top[top_idx]->mutable_cpu_data();
//
//for (int n = 0; n < num_; ++n) {
//// im2col transformation: unroll input regions for filtering
//// into column matrix for multplication.
//if (!is_1x1_) {
//im2col_3d_cpu(
//        bottom_data + bottom[bottom_idx]->offset(n),
//        channels_, height_, width_,
//        block_c_, block_h_, block_w_,
//        pad_c_, pad_h_, pad_w_,
//        stride_c_, stride_h_, stride_w_,
//        col_buff,
//        blocks_round_down_, blocks_out_of_bounds_value_);
//} else {  // special case for 1x1 convolution
//if (!normalize_patches_) {
//col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
//} else {
//caffe_copy(N_ * K_, bottom[bottom_idx]->cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
//}
//}
//if (normalize_patches_) {
//caffe_cpu_transpose(K_, N_,
//        col_buff,
//        row_buffer_.mutable_cpu_data());
//caffe_cpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
//        row_buffer_.mutable_cpu_data(), normalize_variance_);
//caffe_cpu_transpose(N_, K_,
//        row_buffer_.cpu_data(),
//        col_buff);
//}
//// Prepare input
//Dtype* current_top = top_data + top[top_idx]->offset(n);
//if (num_regions_ > 1) {
//split_patches_in = split_patches_in_.mutable_cpu_data();
//split_patches_out = split_patches_out_.mutable_cpu_data();
//split_patches_cpu<Dtype, false>(N_, K_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        col_buff, split_patches_in, use_unshared_regions_);
//} else {
//split_patches_in = col_buff;
//split_patches_out = current_top;
//}
//
//// Calculate
//mex_forward_cpu<Dtype>(M_, region_size_, K_, softmax_mode_, epsilon,
//        offsets, split_patches_in, split_patches_out, num_regions_);
//// Copy to output if needed
//if (num_regions_ > 1) {
//split_patches_cpu<Dtype, true>(N_, M_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        current_top, split_patches_out, use_unshared_regions_);
//}
//}
//}
//if (!use_log_space_parameters_) {
//caffe_exp<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data());
//caffe_cpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data(), linear_space_min_value_);
//}
//}
//
//template <typename Dtype>
//void MEXLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//
//Dtype* split_patches_in = NULL;
//Dtype* split_patches_in_diff = NULL;
//Dtype* split_patches_out = NULL;
//Dtype* split_patches_out_diff = NULL;
//typename vec<Dtype>::vec2* split_patches_out_inter = NULL;
//const Dtype epsilon = this->blobs_[0]->cpu_data()[0];
//Dtype epsilon_diff = 0;
//Dtype* epsilon_helper = NULL;
//if (this->param_propagate_down_[0]) {
//epsilon_helper = static_cast<Dtype*>(epsilon_helper_->mutable_cpu_data());
//}
//
//const Dtype* offsets = this->blobs_[1]->cpu_data();
//if (!use_log_space_parameters_) {
//caffe_cpu_clip_min<Dtype>(num_regions_ * M_ * K_, offsets, this->blobs_[1]->mutable_cpu_data(), linear_space_min_value_);
//caffe_log<Dtype>(num_regions_ * M_ * K_, offsets, this->blobs_[1]->mutable_cpu_data());
//}
//if (normalize_offsets_) {
//mex_forward_cpu<Dtype>(M_ * num_regions_, 1, K_, softmax_mode_, epsilon,
//offsets, one_zero_vec_.cpu_diff(), offsets_norm_factor_.mutable_cpu_data());
//Dtype* offsets_mutable = NULL;
//if (!normalize_offsets_projected_) {
//caffe_copy<Dtype>(num_regions_ * M_ * K_, offsets, normed_offsets_.mutable_cpu_data());
//offsets = normed_offsets_.cpu_data();
//offsets_mutable = normed_offsets_.mutable_cpu_data();
//} else {
//offsets_mutable = this->blobs_[1]->mutable_cpu_data();
//}
//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_ * num_regions_, K_, 1,
//-1, offsets_norm_factor_.cpu_data(), one_zero_vec_.cpu_data(),
//1, offsets_mutable);
//}
//Dtype* offsets_diff = NULL;
//if (this->param_propagate_down_[1]) {
//if (use_log_space_parameters_) {
//offsets_diff = this->blobs_[1]->mutable_cpu_diff();
//} else {
//offsets_diff = normed_offsets_.mutable_cpu_diff();
//}
//}
//
//bool propagate_down_any = false;
//for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
//if (propagate_down[top_idx]) {
//propagate_down_any = true;
//break;
//}
//}
//const Dtype* transposed_offsets = NULL;
//if (propagate_down_any) {
//transposed_offsets = static_cast<const Dtype*>(transposed_offsets_->cpu_data());
//for (int r = 0; r < num_regions_; ++r) {
//const int offsets_idx = r * M_ * K_;
//caffe_cpu_transpose(M_, K_,
//        offsets + offsets_idx,
//static_cast<Dtype*>(transposed_offsets_->mutable_cpu_data()) + offsets_idx);
//}
//}
//
//for (int top_idx = 0; top_idx < top.size(); ++top_idx) {
//const int bottom_idx = expects_labels_ ? top_idx * 2 : top_idx;
//if (this->param_propagate_down_[0] ||
//this->param_propagate_down_[1] ||
//propagate_down[top_idx]) {
//const Dtype* top_diff = top[top_idx]->cpu_diff();
//const Dtype* top_data = top[top_idx]->cpu_data();
//Dtype* col_buff = NULL;
//Dtype* col_diff = NULL;
//if (!is_1x1_ || normalize_patches_) {
//col_buff = col_buffer_.mutable_cpu_data();
//}
//if (!is_1x1_) {
//col_diff = col_buffer_.mutable_cpu_diff();
//}
//const Dtype* bottom_data = bottom[bottom_idx]->cpu_data();
//Dtype* bottom_diff = bottom[bottom_idx]->mutable_cpu_diff();
//for (int n = 0; n < num_; ++n) {
//// Since we saved memory in the forward pass by not storing all col
//// data, we will need to recompute them.
//if (!is_1x1_) {
//im2col_3d_cpu(
//        bottom_data + bottom[bottom_idx]->offset(n),
//        channels_, height_, width_,
//        block_c_, block_h_, block_w_,
//        pad_c_, pad_h_, pad_w_,
//        stride_c_, stride_h_, stride_w_,
//        col_buff,
//        blocks_round_down_, blocks_out_of_bounds_value_);
//} else {  // special case for 1x1 convolution
//col_diff = bottom_diff + bottom[bottom_idx]->offset(n);
//if (!normalize_patches_) {
//col_buff = bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n);
//} else {
//caffe_copy(N_ * K_, bottom[bottom_idx]->mutable_cpu_data() + bottom[bottom_idx]->offset(n), col_buff);
//}
//}
//if (normalize_patches_) {
//caffe_cpu_transpose(K_, N_,
//        col_buff,
//        row_buffer_.mutable_cpu_data());
//caffe_copy(K_ * N_,
//row_buffer_.cpu_data(),
//        row_buffer_.mutable_cpu_diff());
//caffe_cpu_normalize_patches_rows_forward(K_, N_, normalization_fudge_factor_,
//        row_buffer_.mutable_cpu_data(), normalize_variance_);
//caffe_cpu_transpose(N_, K_,
//        row_buffer_.cpu_data(),
//        col_buff);
//}
//// Prepare input for backprop
//const Dtype* current_top_data = top_data + n * M_ * N_;
//const Dtype* current_top_diff = top_diff + n * M_ * N_;
//if (num_regions_ > 1) {
//split_patches_in = split_patches_in_.mutable_cpu_data();
//split_patches_in_diff = split_patches_in_.mutable_cpu_diff();
//split_patches_out = split_patches_out_.mutable_cpu_data();
//split_patches_out_diff = split_patches_out_.mutable_cpu_diff();
//split_patches_cpu<Dtype, false>(N_, K_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        col_buff, split_patches_in, use_unshared_regions_);
//split_patches_cpu<Dtype, false>(N_, M_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        current_top_data, split_patches_out, use_unshared_regions_);
//split_patches_cpu<Dtype, false>(N_, M_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        current_top_diff, split_patches_out_diff, use_unshared_regions_);
//} else {
//split_patches_in = col_buff;
//split_patches_in_diff = col_diff;
//split_patches_out = (Dtype*)current_top_data;
//split_patches_out_diff = (Dtype*)current_top_diff;
//}
//split_patches_out_inter = static_cast<typename vec<Dtype>::vec2 *>(
//        split_patches_out_inter_->mutable_cpu_data());
//interlace_cpu(num_regions_ * M_ * region_size_, split_patches_out, split_patches_out_diff,
//        split_patches_out_inter);
//// Caculate backprop
//if ((this->param_propagate_down_[0] && std::isfinite(epsilon)) || this->param_propagate_down_[1]) {
//// temp use of split_patches_in_diff for transposing the patches
//for (int r = 0; r < num_regions_; ++r) {
//const int input_idx = r * K_ * region_size_;
//caffe_cpu_transpose(K_, region_size_, split_patches_in + input_idx, split_patches_in_diff + input_idx);
//}
//}
//if (this->param_propagate_down_[0] && std::isfinite(epsilon)) { // epsilon = ±inf => epsilon_diff = 0
//if (!normalize_offsets_ || normalize_offsets_projected_) {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
//                mex_backward_epsilon<Dtype>, ggemm_add<Dtype>, false, no_op<Dtype, Dtype>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, epsilon_helper, 0, epsilon, num_regions_);
//} else {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, Dtype,
//                mex_backward_epsilon_with_normalized_offsets<Dtype>, ggemm_add<Dtype>, false,
//                no_op<Dtype, Dtype>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, epsilon_helper, 0, epsilon, num_regions_);
//}
//// For GPU: thrust::device_ptr<float> cptr = thrust::device_pointer_cast(c);
//const Dtype sum_offsets_diff = thrust::reduce(epsilon_helper, epsilon_helper + num_regions_ * M_ * K_);
//epsilon_diff += sum_offsets_diff / (epsilon * K_);
//}
//if (this->param_propagate_down_[1]) {
//if (!use_log_space_parameters_) {
//caffe_set(M_ * K_ * num_regions_, Dtype(0), offsets_diff);
//}
//if (!normalize_offsets_ || normalize_offsets_projected_) {
//if (std::isfinite(epsilon)) {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, typename vec<Dtype>::vec2,
//                mex_backward_offsets_finite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, typename vec<Dtype>::vec2>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, offsets_diff, 0,
//make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
//} else {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
//                mex_backward_offsets_infinite<Dtype>, ggemm_add<Dtype>, true, no_op<Dtype, uint8_t>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, offsets_diff, 0, 0, num_regions_);
//}
//} else {
//if (std::isfinite(epsilon)) {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, typename vec<Dtype>::vec2,
//                mex_backward_normalized_offsets_finite<Dtype>, ggemm_add<Dtype>, true,
//                no_op<Dtype, typename vec<Dtype>::vec2>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, offsets_diff, 0,
//make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
//} else {
//ggemm_readc_cpu
//        <false, false, typename vec<Dtype>::vec2, Dtype, Dtype, uint8_t,
//                mex_backward_normalized_offsets_infinite<Dtype>, ggemm_add<Dtype>,
//                true, no_op<Dtype, uint8_t>, false,
//                true, true, true>
//(M_, K_, region_size_, split_patches_out_inter, split_patches_in_diff,
//        offsets, offsets_diff, 0, 0, num_regions_);
//}
//}
//}
//if (propagate_down[top_idx]) {
//if (std::isfinite(epsilon)) {
//ggemm_readc_cpu
//        <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, typename vec<Dtype>::vec2,
//                mex_backward_bottom_finite<Dtype>, ggemm_add<Dtype>, false,
//                no_op<Dtype, typename vec<Dtype>::vec2>, false,
//                true, true, true>
//(K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
//        split_patches_in, split_patches_in_diff, 0,
//make_vec2<Dtype>(epsilon, softmax_mode_ ? Dtype(0) : (Dtype)-std::log(K_)), num_regions_);
//} else {
//ggemm_readc_cpu
//        <false, false, Dtype, typename vec<Dtype>::vec2, Dtype, uint8_t,
//                mex_backward_bottom_infinite<Dtype>, ggemm_add<Dtype>, false,
//                no_op<Dtype, uint8_t>, false,
//                true, true, true>
//(K_, region_size_, M_, transposed_offsets, split_patches_out_inter,
//        split_patches_in, split_patches_in_diff, 0, 0, num_regions_);
//}
//}
//// Copy to bottom if needed
//if (num_regions_ > 1) {
//split_patches_cpu<Dtype, true>(N_, K_,
//        width_out_, height_out_, channels_out_,
//        offsets_w_, offsets_h_, offsets_c_,
//        shared_offsets_region_w_, shared_offsets_region_h_, shared_offsets_region_c_,
//        col_diff, split_patches_in_diff, use_unshared_regions_);
//}
//
//// Backprop for patch normalization
//if (normalize_patches_ && propagate_down[top_idx]) {
//caffe_cpu_transpose(K_, N_, col_diff, col_buff);
//caffe_cpu_normalize_patches_rows_backward(K_, N_, normalization_fudge_factor_,
//        row_buffer_.cpu_diff(), row_buffer_.cpu_data(), col_buff, normalize_variance_);
//caffe_cpu_transpose(N_, K_, col_buff, col_diff);
//}
//
//if (propagate_down[top_idx] && !is_1x1_) {
//col2im_3d_cpu(
//        col_diff,
//        channels_, height_, width_,
//        block_c_, block_h_, block_w_,
//        pad_c_, pad_h_, pad_w_,
//        stride_c_, stride_h_, stride_w_,
//        bottom_diff + bottom[bottom_idx]->offset(n),
//        blocks_round_down_);
//}
//if (!use_log_space_parameters_ && this->param_propagate_down_[1]) {
//const Dtype* original_logspace_offsets = this->blobs_[1]->cpu_data();
//Dtype* original_offsets_diff = this->blobs_[1]->mutable_cpu_diff();
//for (int i = 0; i < num_regions_ * M_ * K_; ++i) {
//original_offsets_diff[i] +=  offsets_diff[i] / std::max(std::exp(original_logspace_offsets[i]), linear_space_min_value_);
//}
//}
//}
//}
//}
//if (this->param_propagate_down_[0]) {
//this->blobs_[0]->mutable_cpu_diff()[0] += epsilon_diff;
//}
//if (this->param_propagate_down_[1] && this->maximum_entropy_regularization_coeff_ > Dtype(0)) {
//caffe_cpu_maximum_entropy_regularization(num_regions_ * M_, K_, offsets, normed_offsets_.mutable_cpu_diff());
//caffe_axpy(num_regions_ * M_ * K_, maximum_entropy_regularization_coeff_, normed_offsets_.cpu_diff(), offsets_diff);
//}
//if (!use_log_space_parameters_) {
//caffe_exp<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data());
//caffe_cpu_clip_min<Dtype>(num_regions_ * M_ * K_, this->blobs_[1]->cpu_data(), this->blobs_[1]->mutable_cpu_data(), linear_space_min_value_);
//}
//}
//
//#ifdef CPU_ONLY
//STUB_GPU(MEXLayer);
//#endif
//
//INSTANTIATE_CLASS(MEXLayer);
//REGISTER_LAYER_CLASS(MEX);
//}  // namespace caffe
