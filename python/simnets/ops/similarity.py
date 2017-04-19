import tensorflow as _tf

import os as _os
_this_path = _os.path.split(__file__)[0]

_so = _tf.load_op_library(_os.path.join(_this_path, 'simnet_ops.so'))

_similarity = _so.similarity
_similarity_ref = _so.similarity_ref
_similarity_input_grad = _so.similarity_input_grad
_similarity_parameters_grad = _so.similarity_parameters_grad

similarity = _similarity


@_tf.RegisterGradient("Similarity")
def _similarity_grad(op, grad):
    inp = op.inputs[0]
    templates = op.inputs[1]
    weights = op.inputs[2]

    padding = op.get_attr('padding')
    strides = op.get_attr('strides')
    ksize = op.get_attr('ksize')
    similarity_function = op.get_attr('similarity_function')
    normalization_term = op.get_attr('normalization_term')
    normalization_term_fudge = op.get_attr('normalization_term_fudge')
    ignore_nan_input = op.get_attr('ignore_nan_input')
    out_of_bounds_value = op.get_attr('out_of_bounds_value')

    grad_input = _similarity_input_grad(inp, templates, weights, grad, padding=padding, ksize=ksize, strides=strides,
                                        similarity_function=similarity_function,
                                        normalization_term=normalization_term,
                                        normalization_term_fudge=normalization_term_fudge,
                                        ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value)
    grad_templates, grad_weights = _similarity_parameters_grad(inp, templates, weights, grad, padding=padding,
                                                               ksize=ksize, strides=strides,
                                                               similarity_function=similarity_function,
                                                               normalization_term=normalization_term,
                                                               normalization_term_fudge=normalization_term_fudge,
                                                               ignore_nan_input=ignore_nan_input,
                                                               out_of_bounds_value=out_of_bounds_value)
    return [grad_input, grad_templates, grad_weights]
