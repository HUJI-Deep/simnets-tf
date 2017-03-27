import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

so = tf.load_op_library('./simnet_ops.so')
similarity = so.similarity
similarity_input_grad = so.similarity_input_grad
similarity_parameters_grad = so.similarity_parameters_grad
similarity_ref = so.similarity_ref

@tf.RegisterGradient("Similarity")
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

    grad_input = similarity_input_grad(inp, templates, weights, grad, padding=padding, ksize=ksize, strides=strides,
                                       similarity_function=similarity_function,
                                       normalization_term=normalization_term, normalization_term_fudge=normalization_term_fudge,
                                       ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value)
    grad_templates, grad_weights = similarity_parameters_grad(inp, templates, weights, grad, padding=padding, ksize=ksize, strides=strides,
                                                              similarity_function=similarity_function,
                                                              normalization_term=normalization_term, normalization_term_fudge=normalization_term_fudge,
                                                              ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value)
    return [grad_input, grad_templates, grad_weights]

def create_network():


def main():
    net = create_network()

    for i in range(200):
        pass


if __name__ == '__main__':
    main()