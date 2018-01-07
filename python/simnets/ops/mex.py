from __future__ import print_function
from __future__ import division

import tensorflow as _tf
import ctypes as _ctypes

import os as _os
from functools import wraps as _wraps

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


_mex = _so.mex
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
    blocks = op.get_attr('blocks')
    blocks_out_of_bounds_value = op.get_attr('blocks_out_of_bounds_value')
    blocks_round_down = op.get_attr('blocks_round_down')
    use_unshared_regions = op.get_attr('use_unshared_regions')
    shared_offset_region = op.get_attr('shared_offset_region')
    unshared_offset_region = op.get_attr('unshared_offset_region')

    grad_input = _mex_input_grad(inp, offsets, output, grad, num_instances=num_instances, softmax_mode=softmax_mode,
                                 padding=padding, strides=strides, epsilon=epsilon,
                                 blocks=blocks,
                                 blocks_out_of_bounds_value=blocks_out_of_bounds_value,
                                 blocks_round_down=blocks_round_down,
                                 use_unshared_regions=use_unshared_regions,
                                 shared_offset_region=shared_offset_region,
                                 unshared_offset_region=unshared_offset_region)

    grad_offsets = _mex_offsets_grad(inp, offsets, output, grad, num_instances=num_instances, softmax_mode=softmax_mode,
                                     padding=padding, strides=strides, epsilon=epsilon,
                                     blocks=blocks,
                                     blocks_out_of_bounds_value=blocks_out_of_bounds_value,
                                     blocks_round_down=blocks_round_down, use_unshared_regions=use_unshared_regions,
                                     shared_offset_region=shared_offset_region,
                                     unshared_offset_region=unshared_offset_region)
    return [grad_input, grad_offsets]


def _expand_dim_specification(image_shape, dim_spec):
    """Expand mex dimension specification.

    The dimension specification can be 2 or 3 long, it is
    processed in two steps:
    1. If it is of length 2, a -1 is prepended to it
    2. Each dimension with -1 is replaced with the whole corresponding image dimension

    Args:
        image_shape : list(int)
            the shape of the input image, of length 3 (without batch) or 4 (with bach)
        dim_spec : list(int)
            the specification to be expanded

    Returns:
        The expanded dimension specification
    """
    if len(dim_spec) != 2 and len(dim_spec) != 3:
        raise ValueError('Bad dimensions specifications, should be a list of two or three, got %s' % dim_spec)
    if len(image_shape) == 3:  # we make sure image dimensions of length 4 (includes batch)
        image_shape = [None] + image_shape
    if len(dim_spec) == 2:
        dim_spec = [-1] + list(dim_spec)
    dim_spec = dim_spec[:]  # copy, not to step on outer scope values
    for i in range(3):
        if dim_spec[i] == -1:
            dim_spec[i] = image_shape[i + 1]  # +1 for batch dimension
    return dim_spec


def mex(input, offsets, num_instances, softmax_mode=None, padding=None, strides=None, blocks=None, epsilon=None,
        blocks_out_of_bounds_value=None, blocks_round_down=None, use_unshared_regions=None, shared_offset_region=None,
        unshared_offset_region=None, name=None):
    r"""Computes the MEX layer given 4-D `input` and 5-D `offsets` tensors.

  As defined in https://arxiv.org/abs/1506.03059

  Given an input tensor of shape `[batch, in_channels, in_height, in_width]`
  and a offsets tensor of shape
  `[num_regions, num_instances, filter_channels, filter_height, filter_width]`,  where
  num_regions is calculated from the output dimensions and the shared/unshared offsets parmaeter

  This op performs the following:
  Extract virtual patches of size `blocks` from the input tensor,
  according to the `padding`, `strides` and `blocks` parameters.
  this results in a 3D grid of patches indexed by c,i,j.
  For each output element we select the corresponding patch and offsets region
  then calculate:

  .. math::

            \frac{1}{\epsilon} \log\left(\frac{1}{n} \sum\exp(\epsilon (patch + region))\right)

  The different parameters change the behaviour as described below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A 4-D tensor. with dimensions `[batch, in_channels, in_height, in_width]`.
    offsets: A `Tensor`. Must have the same type as `input`.
      A 5-D tensor of shape
      `[num_regions, num_instances, filter_channels, filter_height, filter_width]`
      must be non negative!
    num_instances: An `int`. the number of instances of the layer.
    softmax_mode: An optional `bool`. Defaults to `False`.
      in softmax mode we do not divide by the patch size inside of the log
    padding: An optional list of `ints`. Defaults to `[0, 0, 0]`.
      list of length 3.  The padding to use
      for the dimensions of `input`.
    strides: An optional list of `ints`. Defaults to `[1, 1, 1]`.
      list of length 3.  The stride of the sliding window
      for the dimensions of `input`.
    blocks: An optional list of `ints`. Defaults to `[1, 1, 1]`.
      list of length 3.  The 3D dimensions of the blocks.
    epsilon: An optional `float`. Defaults to `1`.
      the epsilon parameter. can be +inf, -inf
    blocks_out_of_bounds_value: An optional `float`. Defaults to `0`.
      value to use for out of bounds elements
    blocks_round_down: An optional `bool`. Defaults to `True`.
      controls the calculation of the output size.
      with round_down it is::

          image_size + 2 * pad_size - patch_size) / stride + 1

      without it is::

          static_cast<int>(
             std::ceil(static_cast<float>(
                 image_size + 2 * pad_size - patch_size) / stride)) + 1
    use_unshared_regions: An optional `bool`. Defaults to `True`.
      alternative to defining a shared region, unshared region.
    shared_offset_region: An optional list of `ints`. Defaults to `[-1]`.
      the region in which offsets are shared.
      a value of -1 is replaced by the entire respective dimension.
      can be a list of length 3, or 1. if it is of length 1 [d], it is
      expanded to [-1, d, d]
    unshared_offset_region: An optional list of `ints`. Defaults to `[-1]`.
      the region in which offsets are unshared.
      a value of -1 is replaced by the entire respective dimension.
      can be a list of length 3, or 1. if it is of length 1 [d], it is
      expanded to [-1, d, d]
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. A 4-D tensor of shape
    `[batch, out_channels, out_height, out_width]`
"""
    return _mex(input, offsets, num_instances, softmax_mode=softmax_mode, padding=padding, strides=strides,
                blocks=blocks,
                epsilon=epsilon,
                blocks_out_of_bounds_value=blocks_out_of_bounds_value, blocks_round_down=blocks_round_down,
                use_unshared_regions=use_unshared_regions, shared_offset_region=shared_offset_region,
                unshared_offset_region=unshared_offset_region, name=name)
