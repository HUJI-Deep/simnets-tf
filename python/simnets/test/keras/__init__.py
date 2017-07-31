from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import keras
import keras.models

from ...keras import Similarity, Mex

class KerasMexTests(tf.test.TestCase):

    def test_attributes(self):
        l = Mex(num_instances=10, padding='VALID',
                strides=[1, 2, 2], blocks=[1, 2, 3], epsilon=2.0,
                use_unshared_regions=False,shared_offset_region=[2, 2],
                unshared_offset_region=[4, 4], softmax_mode=True, blocks_out_of_bounds_value=1.0,
                blocks_round_down=False, normalize_offsets=True, input_shape=(3, 40, 40))
        model = keras.models.Sequential([l])
        model.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
        op = l.op.op  # type: tf.Operation
        self.assertTrue(op.get_attr('num_instances') == 10)
        self.assertTrue(op.get_attr('padding') == [0, 0, 0])
        self.assertTrue(op.get_attr('strides') == [1, 2, 2])
        self.assertTrue(op.get_attr('blocks') == [1, 2, 3])
        self.assertTrue(op.get_attr('epsilon') == 2.0)
        self.assertTrue(op.get_attr('use_unshared_regions') is False)
        self.assertTrue(op.get_attr('shared_offset_region') == [2, 2])
        self.assertTrue(op.get_attr('unshared_offset_region') == [4, 4])
        self.assertTrue(op.get_attr('softmax_mode') is True)
        self.assertTrue(op.get_attr('blocks_out_of_bounds_value') == 1.0)
        self.assertTrue(op.get_attr('blocks_round_down') is False)
        self.assertTrue('NormalizeOffsets' in op.inputs[1].name)

    def _run_shape_test(self, num_instances, padding, strides, blocks, blocks_round_down,
                        input_shape, output_shape):
        l = Mex(num_instances=num_instances, padding=padding,
                strides=strides, blocks=blocks, epsilon=2.0,
                use_unshared_regions=False, shared_offset_region=[2, 2],
                unshared_offset_region=[4, 4], softmax_mode=True, blocks_out_of_bounds_value=1.0,
                blocks_round_down=blocks_round_down, normalize_offsets=True, input_shape=input_shape)
        model = keras.models.Sequential([l])
        model.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
        output = l.op  # type: tf.Tensor
        self.assertEqual(output.get_shape().as_list()[1:], list(output_shape))

    def test_shapes(self):
        self._run_shape_test(10, [1, 0, 1], [1, 1, 1], [1, 2, 2], False,
                             (3, 10, 13),
                             (50, 9, 14))
        self._run_shape_test(10, [2, 1, 1], [1, 3, 2], [1, 2, 2], False,
                             (3, 10, 13),
                             (70, 5, 8))
        self._run_shape_test(2, [1, 1, 1], [1, 3, 2], [1, 2, 2], False,
                             (1, 10, 13),
                             (6, 5, 8))

    def test_channels_last(self):
        with self.test_session():
            l1 = Mex(num_instances=3, padding='VALID',
                     strides=[1,2,2], blocks=[1,1,1], epsilon=1.0,
                     use_unshared_regions=True, unshared_offset_region=[2], softmax_mode=True,
                     offsets_initializer='ones', input_shape=(3, 10, 11))
            l2 = Mex(num_instances=3, padding='VALID',
                     strides=[1,2,2], blocks=[1, 1, 1], epsilon=1.0,
                     offsets_initializer='ones', use_unshared_regions=True,
                     unshared_offset_region=[2],
                     softmax_mode=True, channels_last=True, input_shape=(10, 11, 3))
            model1 = keras.models.Sequential([l1])
            model2 = keras.models.Sequential([l2])
            model1.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
            model2.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))

            data = np.random.normal(size=(1, 3, 10, 11))
            data_t = np.transpose(data, [0, 2, 3, 1])
            res1 = model1.predict(data)
            res2 = model2.predict(data_t)
            res2_t = np.transpose(res2, [0, 3, 1, 2])
            self.assertAllClose(res1, res2_t)


class KerasSimilarityTests(tf.test.TestCase):

    def test_attributes(self):
        l = Similarity(num_instances=5, similarity_function='L1',
                       strides=[2, 3], blocks=[3, 2], padding='VALID',
                       normalization_term=False, normalization_term_fudge=1.0,
                       ignore_nan_input=True, out_of_bounds_value=1.0, input_shape=(3, 40, 40))
        model = keras.models.Sequential([l])
        model.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
        op = l.op.op  # type: tf.Operation
        self.assertEqual(op.get_attr('blocks'), [3, 2])
        self.assertEqual(op.get_attr('strides'), [2, 3])
        self.assertEqual(op.get_attr('padding'), [0, 0])
        self.assertEqual(op.get_attr('normalization_term'), False)
        self.assertEqual(op.get_attr('ignore_nan_input'), True)
        self.assertEqual(op.get_attr('similarity_function'), b'L1')
        self.assertNear(op.get_attr('normalization_term_fudge'), 1.0, 0.001)
        self.assertEqual(op.get_attr('out_of_bounds_value'), 1.0)

    def _run_shape_test(self, num_instances, padding, strides, blocks, input_shape, output_shape):
        l = Similarity(num_instances=num_instances, similarity_function='L1',
                       strides=strides, blocks=blocks, padding=padding,
                       normalization_term=False, normalization_term_fudge=1.0,
                       ignore_nan_input=True, out_of_bounds_value=1.0, input_shape=input_shape)
        model = keras.models.Sequential([l])
        model.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
        output = l.op  # type: tf.Tensor
        self.assertEqual(output.get_shape().as_list()[1:], list(output_shape))

    def test_shapes(self):
        self._run_shape_test(10, [0, 1], [1, 1], [2, 2],
                             (3, 10, 13),
                             (10, 9, 14))
        self._run_shape_test(10, [1, 1], [3, 2], [2, 2],
                             (3, 10, 13),
                             (10, 4, 7))
        self._run_shape_test(2, [1, 1], [3, 2], [2, 2],
                             (1, 10, 13),
                             (2, 4, 7))

    def test_channels_last(self):
        with self.test_session():
                l1 = Similarity(num_instances=3, padding='VALID',
                                strides=[2, 2], blocks=[1, 1],
                                templates_initializer='ones', weights_initializer='ones', input_shape=(3, 10, 11))
                l2 = Similarity(num_instances=3, padding='VALID',
                                strides=[2,2], blocks=[1,1],
                                templates_initializer='ones', weights_initializer='ones',
                                channels_last=True, input_shape=(10, 11, 3))
                model1 = keras.models.Sequential([l1])
                model2 = keras.models.Sequential([l2])
                model1.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))
                model2.compile(optimizer='adam', loss=lambda a, b: tf.constant([0], tf.float32))

                data = np.random.normal(size=(1, 3, 10, 11))
                data_t = np.transpose(data, [0, 2, 3, 1])
                res1 = model1.predict(data)
                res2 = model2.predict(data_t)
                res2_t = np.transpose(res2, [0, 3, 1, 2])
                self.assertAllClose(res1, res2_t)



