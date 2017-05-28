//
// Created by elhanani on 27/03/17.
//
#include "kernels/mex_kernel_common.hpp"

#define DLL_PUBLIC __attribute__ ((visibility ("default")))

extern "C" DLL_PUBLIC int get_mex_offsets_nregions(int, const int* input_dim,
                                                   int n_padding, const int* padding,
                                                   int n_strides, const int* strides,
                                                   int num_instances, int blocks_round_down, int use_unshared_regions,
                                                   int n_blocks, const int* blocks,
                                                   int n_shared_offset_region, const int* shared_offset_region,
                                                   int n_unshared_offset_region, const int* unshared_offset_region)
{
    int input_c = input_dim[0];
    int input_h = input_dim[1];
    int input_w = input_dim[2];
    MexDimensionsData mdd;
    mdd.padding_.assign(padding, padding + n_padding);
    mdd.strides_.assign(strides, strides + n_strides);
    mdd.shared_offset_region_.assign(shared_offset_region, shared_offset_region + n_shared_offset_region);
    mdd.unshared_offset_region_.assign(unshared_offset_region, unshared_offset_region + n_unshared_offset_region);

    if (n_blocks != 3) {
        return -1;
    }
    mdd.blocks_.assign(blocks, blocks + n_blocks);
    //mdd.block_c_ = blocks[0];
    //mdd.block_h_ = blocks[1];
    //mdd.block_w_ = blocks[2];


    mdd.batch_ = 1;
    mdd.channels_ = input_c;
    mdd.height_ = input_h;
    mdd.width_ = input_w;
    mdd.num_instances_ = num_instances;
    mdd.blocks_round_down_ = blocks_round_down > 0;
    mdd.use_unshared_regions_ = use_unshared_regions > 0;

    mdd.CalculateDimensions();
    return mdd.num_regions_;
}

