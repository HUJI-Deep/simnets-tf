import tensorflow as tf
import unittest
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import itertools

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

# TODO: Test 1x1 case
# TODO: Test different attributes
# TODO: Test different odd/even same/valid combinations
# TODO: Test float/double


class SimilarityTests(tf.test.TestCase):

    def _run_ref_test(self, tests_dict):
        all_tests = [dict(zip(tests_dict.keys(), v)) for v in itertools.product(*tests_dict.values())]
        for idx, tst in enumerate(all_tests):
            with self.subTest(**tst):
                with self.test_session():
                    images = np.random.normal(size=tst['im_dims']).astype(tst['dtype'])
                    #images = np.ones((1,3,800,800), np.float32)
                    images = tf.constant(images)

                    params_dim = (1,tst['im_dims'][1],tst['ksizes'][1], tst['ksizes'][2])
                    weights = np.absolute(np.random.normal(size=params_dim).astype(tst['dtype']))
                    #weights = np.ones((1,3,3,3), np.float32)
                    weights = tf.constant(weights)

                    templates = np.random.normal(size=params_dim).astype(tst['dtype'])
                    #templates = np.zeros((1,3,3,3), np.float32)
                    templates = tf.constant(templates)
                    args = (images, templates, weights)
                    kwargs = dict(ksize=tst['ksizes'], strides=tst['strides'],
                                  padding=tst['padding'], normalization_term=tst['normalize'], ignore_nan_input=tst['ignore_nan'])
                    with tf.device(tst['device']):
                        sim = similarity(*args, **kwargs)
                    sim_ref = similarity_ref(*args, **kwargs)
                    s = sim.eval()
                    sr = sim_ref.eval()
                    self.assertNDArrayNear(s, sr, 1e-2)

    def test_reference_dimensions(self):
        tests_dict = {'strides': [[1,s1,s2,1] for s1 in [1,2] for s2 in [1,2]],
                      'im_dims': [(a,b,c,d) for a in [1, 2] for b in [1, 3] for c in [40] for d in [40]],
                      'dtype': [np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,s1,s2,1] for s1 in [1,3] for s2 in [1,3]],
                      'padding': ['SAME', 'VALID'],
                      'ignore_nan': [False],
                      'normalize': [False]}
        self._run_ref_test(tests_dict)


    def test_reference_params(self):
        tests_dict = {'strides': [[1,2,2,1]],
                      'im_dims': [[1,3,30,30]],
                      'dtype': [np.float32, np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,3,3,1]],
                      'padding': ['SAME'],
                      'ignore_nan': [True, False],
                      'normalize': [True, False]}
        self._run_ref_test(tests_dict)


    def testGradientInput(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,1,30,30), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float64)
            templates = tf.constant(templates)

            with tf.device('/gpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def testGradientTemplates(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((2,3,40,40), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,3,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates_np = np.zeros((1,3,3,3), np.float64)
            templates = tf.constant(templates_np)

            with tf.device('/gpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,4,4,1], padding='SAME')
                computed, numeric = tf.test.compute_gradient(templates, templates.get_shape().as_list(), tf.abs(sim), [2,1,10,10], delta=1e-2)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def testGradientWeights(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,1,30,30), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float64)
            templates = tf.constant(templates)

            with tf.device('/cpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            computed, numeric = tf.test.compute_gradient(weights, weights.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-4)


if __name__ == '__main__':
    tf.test.main()
