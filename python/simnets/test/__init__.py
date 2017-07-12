from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import itertools
import six
import contextlib

from ..ops.mex import mex, _mex_ref, _mex_dims_helper
from ..ops.similarity import similarity, _similarity_ref
class SimilarityTests(tf.test.TestCase):

    if six.PY2:
        @contextlib.contextmanager
        def subTest(self, **kwargs):
            yield

    def _run_ref_test(self, tests_dict):
        all_tests = [dict(zip(tests_dict.keys(), v)) for v in itertools.product(*tests_dict.values())]
        for idx, tst in enumerate(all_tests):
            with self.subTest(**tst):
                with self.test_session():
                    if tst['padding'] == 'SAME':
                        tst['padding'] = [e//2 for e in tst['ksizes']]
                    else:
                        tst['padding'] = [0, 0]
                    images = np.random.normal(size=tst['im_dims']).astype(tst['dtype'])
                    #images = np.ones((1,3,800,800), np.float32)
                    images = tf.constant(images)

                    params_dim = (1,tst['im_dims'][1],tst['ksizes'][0], tst['ksizes'][1])
                    weights = np.absolute(np.random.normal(size=params_dim).astype(tst['dtype']))
                    #weights = np.ones((1,3,3,3), np.float32)
                    weights = tf.constant(weights)

                    templates = np.random.normal(size=params_dim).astype(tst['dtype'])
                    #templates = np.zeros((1,3,3,3), np.float32)
                    templates = tf.constant(templates)
                    args = (images, templates, weights)
                    kwargs = dict(ksize=tst['ksizes'], strides=tst['strides'],
                                  padding=tst['padding'], normalization_term=tst['normalize'],
                                  ignore_nan_input=tst['ignore_nan'], similarity_function=tst['similarity_function'])
                    with tf.device(tst['device']):
                        sim = similarity(*args, **kwargs)
                    sim_ref = _similarity_ref(*args, **kwargs)
                    s = sim.eval()
                    sr = sim_ref.eval()
                    self.assertNDArrayNear(s, sr, 1e-2)

    def _run_grad_test(self, tests_dict):
        all_tests = [dict(zip(tests_dict.keys(), v)) for v in itertools.product(*tests_dict.values())]
        for idx, tst in enumerate(all_tests):
            with self.subTest(**tst):
                with self.test_session():
                    np.random.seed(167)
                    if tst['padding'] == 'SAME':
                        tst['padding'] = [e//2 for e in tst['ksizes']]
                    else:
                        tst['padding'] = [0, 0]
                    images_np = 16.0 + np.random.normal(size=tst['im_dims']).astype(tst['dtype'])
                    #images_np = np.ones(tst['im_dims'], tst['dtype'])
                    images = tf.constant(images_np)

                    params_dim = (1,tst['im_dims'][1],tst['ksizes'][0], tst['ksizes'][1])
                    weights_np = 3.0 + np.absolute(np.random.normal(size=params_dim).astype(tst['dtype']))
                    #weights_np = np.ones(params_dim, tst['dtype'])
                    weights = tf.constant(weights_np)

                    templates_np = 3.0 + np.random.normal(size=params_dim).astype(tst['dtype'])
                    #templates_np = np.zeros(params_dim, tst['dtype'])
                    templates = tf.constant(templates_np)
                    args = (images, templates, weights)
                    kwargs = dict(ksize=tst['ksizes'], strides=tst['strides'],
                                  padding=tst['padding'], normalization_term=tst['normalize'],
                                  ignore_nan_input=tst['ignore_nan'], similarity_function=tst['similarity_function'])

                    with tf.device(tst['device']):
                        sim = similarity(*args, **kwargs)
                        if tst['kind'] == 'input':
                            computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(),
                                                                         tf.reduce_sum(sim), [1], delta=1e-1,
                                                                         x_init_value=images_np)
                        elif tst['kind'] == 'templates':
                            computed, numeric = tf.test.compute_gradient(templates, templates.get_shape().as_list(),
                                                                         tf.reduce_sum(sim), [1], delta=1e-1,
                                                                         x_init_value=templates_np)
                        elif tst['kind'] == 'weights':
                            computed, numeric = tf.test.compute_gradient(weights, weights.get_shape().as_list(),
                                                                         tf.reduce_sum(sim), [1], delta=1e-1,
                                                                         x_init_value=weights_np)
                        else:
                            raise ValueError("kind must be one of 'input', 'weights' or 'templates")
                        atol = 1e-1 if tst['dtype'] == np.float32 else 1e-3
                        self.assertAllClose(computed, numeric, rtol=1e-2, atol=atol)


    def test_reference_dimensions(self):
        tests_dict = {'strides': [[s1,s2] for s1 in [1,2] for s2 in [1,2]],
                      'im_dims': [(a,b,c,d) for a in [1, 2] for b in [1, 3] for c in [40] for d in [40]],
                      'dtype': [np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[s1,s2] for s1 in [1,3] for s2 in [1,3]],
                      'padding': ['SAME', 'VALID'],
                      'ignore_nan': [False],
                      'normalize': [False],
                      'similarity_function': ['L2']}
        self._run_ref_test(tests_dict)


    def test_reference_params(self):
        tests_dict = {'strides': [[2,2]],
                      'im_dims': [[1,3,30,30]],
                      'dtype': [np.float32, np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[3,3]],
                      'padding': ['SAME'],
                      'ignore_nan': [True, False],
                      'normalize': [True, False],
                      'similarity_function': ['L2']}
        self._run_ref_test(tests_dict)

    def test_reference_l1(self):
        tests_dict = {'strides': [[1,1]],
                      'im_dims': [[1,1,1,1]],
                      'dtype': [np.float32, np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,1]],
                      'padding': ['SAME', 'VALID'],
                      'ignore_nan': [False],
                      'normalize': [False],
                      'similarity_function': ['L1']}
        self._run_ref_test(tests_dict)

    def test_grads(self):
        tests_dict = {'strides': [[1,1], [2,2]],
                      'im_dims': [[1,3,10,10]],
                      'dtype': [np.float64, np.float32],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,1], [3,3]],
                      'padding': ['VALID'],
                      'ignore_nan': [False, True],
                      'normalize': [True],
                      'similarity_function': ['L2', 'L1'],
                      'kind': ['weights', 'templates', 'input']}
        self._run_grad_test(tests_dict)


def dirichlet_init(shape):
    num_regions, num_instances, block_c, block_h, block_w = shape
    k = block_c * block_h * block_w
    # when given s as a size argument dirichlet function return an array with shape s + [k]
    # then we reshape the output to be of the same shape as the variable
    init_np = np.random.dirichlet([1.0] * k, size=(num_regions, num_instances))
    init_np = np.log(init_np)
    init_np = init_np.reshape(shape)
    return init_np

class MexTests(tf.test.TestCase):

    if six.PY2:
        @contextlib.contextmanager
        def subTest(self, **kwargs):
            yield

    def test_sanity(self):
        #images = np.zeros((1,1,9,9), np.float64)
        images = np.random.normal(size=(5,1,90,90)).astype(np.float64)
        images = tf.constant(images)

        nregions = _mex_dims_helper([1, 90, 90], 3, blocks=[1,3,3], padding=[0, 1, 1], strides=[1, 2, 2])
        #offsets = np.ones((nregions, 3, 1, 3, 3), np.float64)
        offsets = np.random.normal(size=(nregions, 3, 1, 3, 3)).astype(np.float64)
        offsets = tf.constant(offsets)

        with tf.device('/cpu:0'):
            m = mex(images, offsets, num_instances=3, epsilon=1, blocks=[1, 3, 3], padding=[0, 1, 1], strides=[1, 2, 2])
        mr = _mex_ref(images, offsets, num_instances=3, epsilon=1, blocks=[1, 3, 3], padding=[0, 1, 1], strides=[1, 2, 2])
        with self.test_session():
            mnp = m.eval()
            mrnp = mr.eval()
        self.assertNDArrayNear(mnp, mrnp, 1e-2)

    def _run_ref_test(self, tests_dict):
        all_tests = [dict(zip(tests_dict.keys(), v)) for v in itertools.product(*tests_dict.values())]
        for idx, tst in enumerate(all_tests):
            with self.subTest(**tst):
                with self.test_session():
                    tst['blocks'][0] = min(tst['blocks'][0], tst['im_dims'][1])
                    images = np.random.normal(size=tst['im_dims']).astype(tst['dtype'])
                    #images = np.ones(tst['im_dims'], tst['dtype'])
                    images = tf.constant(images)

                    nregions = _mex_dims_helper(tst['im_dims'][1:], tst['num_instances'], blocks=tst['blocks'],
                                                padding=tst['padding'], strides=tst['strides'],
                                                use_unshared_regions=tst['use_unshared_regions'],
                                                shared_offset_region=tst['shared_offset_region'],
                                                unshared_offset_region=tst['unshared_offset_region'])
                    params_dim = (nregions, tst['num_instances']) + tuple(tst['blocks'])
                    offsets = np.random.normal(size=params_dim).astype(tst['dtype'])
                    #offsets = np.ones(params_dim, tst['dtype'])
                    offsets = tf.constant(offsets)

                    args = (images, offsets)
                    kwargs = dict(strides=tst['strides'],
                                  padding=tst['padding'], num_instances=tst['num_instances'],
                                  softmax_mode=tst['softmax_mode'],
                                  use_unshared_regions=tst['use_unshared_regions'],
                                  shared_offset_region=tst['shared_offset_region'],
                                  unshared_offset_region=tst['unshared_offset_region'])
                    with tf.device(tst['device']):
                        m = mex(*args, **kwargs)
                    m_ref = _mex_ref(*args, **kwargs)
                    mnp = m.eval()
                    mrnp = m_ref.eval()
                    self.assertAllClose(mnp, mrnp, rtol=1e-3, atol=1e-3)

    def _run_grad_test(self, tests_dict, kind='input'):
        all_tests = [dict(zip(tests_dict.keys(), v)) for v in itertools.product(*tests_dict.values())]
        for idx, tst in enumerate(all_tests):
            with self.subTest(**tst):
                with self.test_session() as sess:
                    tst['blocks'][0] = min(tst['blocks'][0], tst['im_dims'][1])
                    images_np = np.random.normal(size=tst['im_dims']).astype(tst['dtype'])
                    #images = np.ones(tst['im_dims'], tst['dtype'])
                    images = tf.constant(images_np)

                    nregions = _mex_dims_helper(tst['im_dims'][1:], tst['num_instances'], blocks=tst['blocks'],
                                                padding=tst['padding'], strides=tst['strides'],
                                                use_unshared_regions=tst['use_unshared_regions'],
                                                shared_offset_region=tst['shared_offset_region'],
                                                unshared_offset_region=tst['unshared_offset_region'])
                    params_dim = (nregions, tst['num_instances']) + tuple(tst['blocks'])
                    offsets_np = -np.abs(np.random.normal(size=params_dim).astype(tst['dtype']))
                    offsets_np = dirichlet_init(params_dim).astype(tst['dtype'])
                    #offsets_np = np.ones(params_dim, tst['dtype'])
                    offsets = tf.constant(offsets_np)

                    args = (images, offsets)
                    kwargs = dict(strides=tst['strides'],
                                  padding=tst['padding'], num_instances=tst['num_instances'],
                                  softmax_mode=tst['softmax_mode'], blocks=tst['blocks'],
                                  use_unshared_regions=tst['use_unshared_regions'],
                                  shared_offset_region=tst['shared_offset_region'],
                                  unshared_offset_region=tst['unshared_offset_region'])
                    atol = 1e-1 if tst['dtype'] == np.float32 else 1e-3
                    with tf.device(tst['device']):
                        m = mex(*args, **kwargs)
                        if kind == 'input':
                            computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(), tf.reduce_sum(m),
                                                                         [1], delta=1e-2, x_init_value=images_np)
                        else:
                            computed, numeric = tf.test.compute_gradient(offsets, offsets.get_shape().as_list(), tf.reduce_sum(m),
                                                                         [1], delta=1e-2, x_init_value=offsets_np)
                    self.assertAllClose(computed, numeric, rtol=1e-2, atol=atol)

    def test_reference_dimensions_shared(self):
        tests_dict = {
            'strides': [[1,s1,s2] for s1 in [1,4] for s2 in [1,4]],
            'im_dims': [(a,b,c,d) for a in [1, 2] for b in [1, 3] for c in [41,40] for d in [40]],
            'dtype': [np.float32],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[s1,s2,s3] for s1 in [1] for s2 in [1,3] for s3 in [3]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [False],
            'shared_offset_region': [[2], [3]],
            'unshared_offset_region': [[1]]}
        self._run_ref_test(tests_dict)

    def test_reference_dimensions_unshared(self):
        tests_dict = {
            'strides': [[1,s1,s2] for s1 in [1,4] for s2 in [1,4]],
            'im_dims': [(a,b,c,d) for a in [1, 2] for b in [1, 3] for c in [41,40] for d in [40]],
            'dtype': [np.float32],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[s1,s2,s3] for s1 in [1] for s2 in [1,3] for s3 in [3]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [True],
            'shared_offset_region': [[2]],
            'unshared_offset_region': [[1,5,4], [2]]}
        self._run_ref_test(tests_dict)

    def test_gradient_input(self):
        tests_dict = {
            'strides': [[1,1,2]],
            'im_dims': [(2,3,12,13)],
            'dtype': [np.float64, np.float32],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[3,3,1]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [True, False],
            'shared_offset_region': [[1,5,4], [-1]],
            'unshared_offset_region': [[1,5,4], [-1]]}
        self._run_grad_test(tests_dict, kind='input')

    def test_gradient_offsets(self):
        tests_dict = {
            'strides': [[1,1,2]],
            'im_dims': [(2,3,12,13)],
            'dtype': [np.float64, np.float32],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[3,3,1]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [True, False],
            'shared_offset_region': [[1,5,4], [2], [-1]],
            'unshared_offset_region': [[1,5,4], [2]]}
        self._run_grad_test(tests_dict, kind='offsets')

    def test_gradient_offsets_blocks_bug(self):
        tests_dict = {
            'strides': [[1,1,2]],
            'im_dims': [(2,3,16,16)],
            'dtype': [np.float64],
            'num_instances': [2],
            'device': ['/gpu:0'],
            'blocks': [[3,8,8]],
            'padding': [[0,4,4]],
            'softmax_mode': [True],
            'use_unshared_regions': [True],
            'shared_offset_region': [[3, 3, 3]],
            'unshared_offset_region': [[2]]}
        self._run_grad_test(tests_dict, kind='offsets')

if __name__ == '__main__':
    tf.test.main()
