from __future__ import print_function
from __future__ import division

import tensorflow as _tf
import ctypes as _ctypes

import os as _os
_this_path = _os.path.split(__file__)[0]

_so = _tf.load_op_library(_os.path.join(_this_path, 'simnet_ops.so'))

_mex_helper = _ctypes.CDLL(_os.path.join(_this_path, 'libmex_dims_helper.so'))


def _mex_dims_helper(input_dim, num_instances,
                    blocks,
                    padding=[0], strides=[1],
                    blocks_round_down=True, use_unshared_regions=True,
                    shared_offset_region=[-1], unshared_offset_region=[-1]):
    ctypes = _ctypes

    args = []
    def add_array(l):
        args.extend([ctypes.c_int(len(l)), (ctypes.c_int * len(l))(*l)])
    add_array(input_dim)
    add_array(padding)
    add_array(strides)
    args.extend([ctypes.c_int(num_instances), ctypes.c_int(blocks_round_down),
                 ctypes.c_int(use_unshared_regions)])
    add_array(blocks)
    add_array(shared_offset_region)
    add_array(unshared_offset_region)

    return _mex_helper.get_mex_offsets_nregions(*args)
mex = _so.mex
_mex_input_grad = _so.mex_input_grad
_mex_offsets_grad = _so.mex_offsets_grad
_mex_ref = _so.mex_ref  # For tests

@_tf.RegisterGradient("Mex")
def _mex_grad(op, grad):
    inp = op.inputs[0]
    offsets = op.inputs[1]
    output = op.outputs[0]

    num_instances = op.get_attr('num_instances')
    softmax_mode = op.get_attr('softmax_mode')
    padding = op.get_attr('padding')
    strides = op.get_attr('strides')
    epsilon = op.get_attr('epsilon')
    blocks_out_of_bounds_value = op.get_attr('blocks_out_of_bounds_value')
    blocks_round_down = op.get_attr('blocks_round_down')
    use_unshared_regions = op.get_attr('use_unshared_regions')
    shared_offset_region = op.get_attr('shared_offset_region')
    unshared_offset_region = op.get_attr('unshared_offset_region')

    grad_input = _mex_input_grad(inp, offsets, output, grad, num_instances=num_instances, softmax_mode=softmax_mode,
                                 padding=padding, strides=strides, epsilon=epsilon,
                                 blocks_out_of_bounds_value=blocks_out_of_bounds_value,
                                 blocks_round_down=blocks_round_down, use_unshared_regions=use_unshared_regions,
                                 shared_offset_region=shared_offset_region, unshared_offset_region=unshared_offset_region)

    grad_offsets = _mex_offsets_grad(inp, offsets, output, grad, num_instances=num_instances, softmax_mode=softmax_mode,
                                     padding=padding, strides=strides, epsilon=epsilon,
                                     blocks_out_of_bounds_value=blocks_out_of_bounds_value,
                                     blocks_round_down=blocks_round_down, use_unshared_regions=use_unshared_regions,
                                     shared_offset_region=shared_offset_region, unshared_offset_region=unshared_offset_region)
    return [grad_input, grad_offsets]