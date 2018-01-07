from __future__ import print_function
from __future__ import division

import tensorflow as _tf

import os as _os

_this_path = _os.path.split(__file__)[0]

_so = _tf.load_op_library(_os.path.join(_this_path, 'simnet_ops.so'))

_similarity = _so.similarity
_similarity_ref = _so.similarity_ref
_similarity_input_grad = _so.similarity_input_grad
_similarity_parameters_grad = _so.similarity_parameters_grad


@_tf.RegisterGradient("Similarity")
def _similarity_grad(op, grad):
    inp = op.inputs[0]
    templates = op.inputs[1]
    weights = op.inputs[2]

    padding = op.get_attr('padding')
    strides = op.get_attr('strides')
    blocks = op.get_attr('blocks')
    similarity_function = op.get_attr('similarity_function')
    normalization_term = op.get_attr('normalization_term')
    normalization_term_fudge = op.get_attr('normalization_term_fudge')
    ignore_nan_input = op.get_attr('ignore_nan_input')
    out_of_bounds_value = op.get_attr('out_of_bounds_value')

    grad_input = _similarity_input_grad(inp, templates, weights, grad, padding=padding, blocks=blocks, strides=strides,
                                        similarity_function=similarity_function,
                                        normalization_term=normalization_term,
                                        normalization_term_fudge=normalization_term_fudge,
                                        ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value)
    grad_templates, grad_weights = _similarity_parameters_grad(inp, templates, weights, grad, padding=padding,
                                                               blocks=blocks, strides=strides,
                                                               similarity_function=similarity_function,
                                                               normalization_term=normalization_term,
                                                               normalization_term_fudge=normalization_term_fudge,
                                                               ignore_nan_input=ignore_nan_input,
                                                               out_of_bounds_value=out_of_bounds_value)
    return [grad_input, grad_templates, grad_weights]


def similarity(input, templates, weights, similarity_function=None, blocks=None, strides=None, padding=None,
               normalization_term=None, normalization_term_fudge=None, ignore_nan_input=None, out_of_bounds_value=None,
               name=None):
    r"""Computes a similarity measure given 4-D `input` `templates` and `weights` tensors.

    As defined in `https://arxiv.org/abs/1506.03059`

    Given an input tensor of shape `[batch, in_channels, in_height, in_width]`
    and a templates, weights tensor of shape
    `[out_channels, in_channels, filter_height, filter_width]`, this op
    performs the following:

    1. Extract virtual patches of size `blocks` from the input tensor,
       according to the `padding`, `strides` and `blocks` parameters.
       block size in the channels dimension is always the number of input channels.
       this results in a 2D grid of patches indexed by i,j
    2. For the simplest version, for output element e = `[b, c, i, j]`, compute
       output[b, c, i ,j] = sum(weights[c] * :math:`phi`(templates[c], patches[i, j]))
       where :math:`phi` is either -|a - b|_1 (l1) or -|a - b|_2 (l2)

    Let :math:`I` be the input image, :math:`T` the temapltes, :math:`W` the weights and :math:`O` the output,
    :math:`p` the padding and :math:`s` the strides then the output element at `[b, c, i, j]` is:

    .. math::

           \sum_{dc, di, dj} T[c, dc, di, dj] \cdot \phi(I[b, dc, s[0] \cdot i + di - p[0],
                                                                  s[1] \cdot j + dj - p[1]],
                                                                  T[c, dc, di, dj])

    the different parameters change the behaviour as described below.

    Args:
      input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        A 4-D tensor. with dimensions `[batch, in_channels, in_height, in_width]`.
      templates: A `Tensor`. Must have the same type as `input`.
        A 4-D tensor of shape
        `[out_channels, in_channels, filter_height, filter_width]`
      weights: A `Tensor`. Must have the same type as `input`.
        A 4-D tensor of shape
        `[out_channels, in_channels, filter_height, filter_width]`
        must be non negative!
      similarity_function: An optional `string` from: `"L1", "L2"`. Defaults to `"L2"`.
      blocks: An optional list of `ints`. Defaults to `[3, 3]`.
        list of length 2.  The height and width of the blocks.
      strides: An optional list of `ints`. Defaults to `[2, 2]`.
        list of length 2.  The stride of the sliding window
        for the height and width dimension of `input`.
      padding: An optional list of `ints`. Defaults to `[0, 0]`.
        list of length 2.  The padding to use
        for the height and width dimension of `input`.
      normalization_term: An optional `bool`. Defaults to `False`.
        if true, add a normalization term to the output, used to make the L2 version
        of this operator into a proper (log) probability measure. the normalization term is
        -0.5 * K * log(2*pi) where K is the total block size, or the number of non-nan
        elements in the block if ignore_nan is on.
      normalization_term_fudge: An optional `float`. Defaults to `0.001`.
        TODO
      ignore_nan_input: An optional `bool`. Defaults to `False`.
        if true, and when using L2 with normalization term compute the probability while
        marginalizing over elements which are nan
      out_of_bounds_value: An optional `float`. Defaults to `0`.
        value to use for elements outside the bounds
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `input`. A 4-D tensor of shape
      `[batch, out_channels, out_height, out_width]`
      """
    return _similarity(input, templates, weights, similarity_function=similarity_function, blocks=blocks,
                       strides=strides,
                       padding=padding,
                       normalization_term=normalization_term, normalization_term_fudge=normalization_term_fudge,
                       ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value,
                       name=name)
