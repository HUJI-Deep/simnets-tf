from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import itertools
import six
import contextlib
import keras
import keras.models

from ...keras import Similarity, Mex

class KerasSimilarityTests(tf.test.TestCase):

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



