import tensorflow as tf
import numpy as np
import itertools

from ..ops.mex import mex, _mex_ref, _mex_dims_helper
from ..ops.similarity import similarity, _similarity_ref
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
                                  padding=tst['padding'], normalization_term=tst['normalize'],
                                  ignore_nan_input=tst['ignore_nan'], similarity_function=tst['similarity_function'])
                    with tf.device(tst['device']):
                        sim = similarity(*args, **kwargs)
                    sim_ref = _similarity_ref(*args, **kwargs)
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
                      'normalize': [False],
                      'similarity_function': ['L2']}
        self._run_ref_test(tests_dict)


    def test_reference_params(self):
        tests_dict = {'strides': [[1,2,2,1]],
                      'im_dims': [[1,3,30,30]],
                      'dtype': [np.float32, np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,3,3,1]],
                      'padding': ['SAME'],
                      'ignore_nan': [True, False],
                      'normalize': [True, False],
                      'similarity_function': ['L2']}
        self._run_ref_test(tests_dict)

    def test_reference_l1(self):
        tests_dict = {'strides': [[1,2,2,1], [1,1,1,1]],
                      'im_dims': [[1,3,30,30]],
                      'dtype': [np.float32, np.float64],
                      'device': ['/cpu:0', '/gpu:0'],
                      'ksizes': [[1,3,3,1], [1,1,1,1]],
                      'padding': ['SAME', 'VALID'],
                      'ignore_nan': [False],
                      'normalize': [False],
                      'similarity_function': ['L1']}
        self._run_ref_test(tests_dict)


    def test_gradient_input_l2(self):
        with self.test_session():
            images = np.random.normal(size=(1,1,30,30)).astype(np.float64)
            #images = np.ones((1,1,30,30), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float64)
            templates = tf.constant(templates)

            with tf.device('/gpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', similarity_function='L2')
                computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def test_gradient_input_l1(self):
        with self.test_session():
            images = np.random.normal(size=(1,1,30,30)).astype(np.float64)
            #images = np.ones((1,1,100,100), np.float64)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float64)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float64)
            templates = tf.constant(templates)

            with tf.device('/gpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', similarity_function='L1')
                computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-2)

    def test_gradient_templates_l2(self):
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
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,4,4,1], padding='SAME',
                                 similarity_function='L2')
                computed, numeric = tf.test.compute_gradient(templates, templates.get_shape().as_list(), tf.abs(sim), [2,1,10,10], delta=1e-2)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def test_gradient_templates_l1(self):
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
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,4,4,1], padding='SAME',
                                 similarity_function='L1')
                computed, numeric = tf.test.compute_gradient(templates, templates.get_shape().as_list(), tf.abs(sim), [2,1,10,10], delta=1e-2)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def test_gradient_weights_l2(self):
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
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',
                                 similarity_function='L2')
            computed, numeric = tf.test.compute_gradient(weights, weights.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-4)

    def test_gradient_weights_l1(self):
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
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME',
                                 similarity_function='L1')
            computed, numeric = tf.test.compute_gradient(weights, weights.get_shape().as_list(), tf.reduce_mean(sim), [1], delta=1e-3)
            self.assertNDArrayNear(numeric, computed, 1e-4)


#def get_mex_offsets_dims(input, num_instances, blocks, padding,

class MexTests(tf.test.TestCase):

    def test_sanity(self):
        #images = np.zeros((1,1,9,9), np.float64)
        images = np.random.normal(size=(5,1,90,90)).astype(np.float64)
        images = tf.constant(images)

        nregions = _mex_dims_helper([1, 90, 90], 3, blocks=[3], padding=[1], strides=[2])
        #offsets = np.ones((nregions, 3, 1, 3, 3), np.float64)
        offsets = np.random.normal(size=(nregions, 3, 1, 3, 3)).astype(np.float64)
        offsets = tf.constant(offsets)

        with tf.device('/cpu:0'):
            m = mex(images, offsets, num_instances=3, epsilon=1, padding=[1], strides=[2])
        mr = _mex_ref(images, offsets, num_instances=3, epsilon=1, padding=[1], strides=[2])
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
                    params_dim = (nregions, tst['num_instances'], *tst['blocks'])
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
                    self.assertAllClose(mnp, mrnp, 1e-2)

    def _run_grad_test(self, tests_dict, kind='input'):
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
                    params_dim = (nregions, tst['num_instances'], *tst['blocks'])
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
                        if kind == 'input':
                            computed, numeric = tf.test.compute_gradient(images, images.get_shape().as_list(), tf.reduce_mean(m), [1], delta=1e-3)
                        else:
                            computed, numeric = tf.test.compute_gradient(offsets, offsets.get_shape().as_list(), tf.reduce_mean(m), [1], delta=1e-3)
                    self.assertAllClose(computed, numeric, 1e-2)

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
            'dtype': [np.float64],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[3,3,1]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [True, False],
            'shared_offset_region': [[1,5,4], [2]],
            'unshared_offset_region': [[1,5,4], [2]]}
        self._run_grad_test(tests_dict, kind='input')

    def test_gradient_offsets(self):
        tests_dict = {
            'strides': [[1,1,2]],
            'im_dims': [(2,3,12,13)],
            'dtype': [np.float64],
            'num_instances': [2],
            'device': ['/cpu:0', '/gpu:0'],
            'blocks': [[3,3,1]],
            'padding': [[0,1,0]],
            'softmax_mode': [True],
            'use_unshared_regions': [True, False],
            'shared_offset_region': [[1,5,4], [2], [-1]],
            'unshared_offset_region': [[1,5,4], [2]]}
        self._run_grad_test(tests_dict, kind='offsets')

if __name__ == '__main__':
    tf.test.main()
