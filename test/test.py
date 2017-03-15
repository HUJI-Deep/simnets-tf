import tensorflow as tf
import numpy as np

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

class SimilarityTests(tf.test.TestCase):

    def testSanity(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,3,800,800), np.float32)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,3,3,3), np.float32)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,3,3,3), np.float32)
            templates = tf.constant(templates)

            with tf.device('/cpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,1,1], padding='SAME', normalization_term=True)
            sim_ref = similarity_ref(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,1,1], padding='SAME', normalization_term=True)
            s = sim.eval()
            sr = sim_ref.eval()
            self.assertAllClose(s, sr, 1e-4)

    def testGradient(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,1,3,3), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float64)
            templates = tf.constant(templates)

            with tf.device('/cpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME', normalization_term=True)
            computed, numeric = tf.test.compute_gradient(weights, weights.get_shape().as_list(), tf.reduce_sum(sim), [1], delta=0.00005)
            self.assertAllClose(computed, numeric, 1e-4)


if __name__ == '__main__':
    tf.test.main()
