from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from keras.engine.topology import Layer
from keras import backend as K
from keras.backend.tensorflow_backend import _initialize_variables
from .ops import similarity as _similarity
from .ops.mex import _mex_dims_helper, mex as _mex
from .unsupervised import similarity_unsupervised_init as _similarity_unsupervised_init


class Similarity(Layer):

    def __init__(self, num_instances, similarity_function='L2', strides=[1,1], ksize=[3,3], padding='SAME',
                 normalization_term=False, normalization_term_fudge=0.001, ignore_nan_input=False,
                 out_of_bounds_value=0, templates_initializer='random_normal', weights_initializer='ones', **kwargs):
        super(Similarity, self).__init__(**kwargs)
        self.num_instances = num_instances
        self.similarity_function = similarity_function
        self.strides = strides
        self.ksize = ksize
        if isinstance(padding, str):
            if padding not in ['SAME', 'VALID']:
                raise ValueError('Padding must be one of SAME, VALID (or a list of ints)')
            self.padding = [0, 0] if padding == 'VALID' else [e//2 for e in ksize]
        else:
            self.padding = padding
        self.normalization_term = normalization_term
        self.normalization_term_fudge = normalization_term_fudge
        self.ignore_nan_input = ignore_nan_input
        self.out_of_bounds_value = out_of_bounds_value
        self.templates_initializer = templates_initializer
        self.weights_initializer = weights_initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sweights = self.add_weight(name='weights',
                                        shape=(self.num_instances, input_shape[1], self.ksize[0], self.ksize[1]),
                                        trainable=True,
                                        initializer=self.weights_initializer)
        self.templates = self.add_weight(name='templates',
                                         shape=(self.num_instances, input_shape[1], self.ksize[0], self.ksize[1]),
                                         initializer=self.templates_initializer,
                                         trainable=True)

        super(Similarity, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.op = _similarity(x, self.templates, tf.abs(self.sweights), similarity_function=self.similarity_function,
                              ksize=self.ksize,
                              strides=self.strides, padding=self.padding,
                              normalization_term=self.normalization_term,
                              normalization_term_fudge=self.normalization_term_fudge,
                              ignore_nan_input=self.ignore_nan_input, out_of_bounds_value=self.out_of_bounds_value)
        return self.op

    def compute_output_shape(self, input_shape):
        return tuple(self.op.get_shape().as_list())


def _dirichlet_init(shape, dtype=None):
    if dtype is None:
        dtype = K.floatx()
    num_regions, num_instances, block_c, block_h, block_w = shape
    k = block_c * block_h * block_w
    # when given s as a size argument dirichlet function return an array with shape s + [k]
    # then we reshape the output to be of the same shape as the variable
    init_np = np.random.dirichlet([1] * k, size=(num_regions, num_instances)).astype(dtype)
    init_np = init_np.reshape(shape)
    return tf.constant(init_np)


class Mex(Layer):

    def __init__(self, num_instances, padding=[0, 0, 0], strides=[1, 1, 1], blocks=[1, 1, 1],
                 epsilon=1.0, use_unshared_regions=True, shared_offset_region=[-1],
                 unshared_offset_region=[-1], softmax_mode=False, blocks_out_of_bounds_value=0.0,
                 blocks_round_down=True, normalize_offsets=False, offsets_initializer=_dirichlet_init, **kwargs):
        super(Mex, self).__init__(**kwargs)
        self.num_instances = num_instances
        self.padding = padding
        self.strides = strides
        self.blocks = blocks
        self.epsilon = epsilon
        self.use_unshared_regions = use_unshared_regions
        self.shared_offset_region = shared_offset_region
        self.unshared_offset_region = unshared_offset_region
        self.softmax_mode = softmax_mode
        self.blocks_out_of_bounds_value = blocks_out_of_bounds_value
        self.blocks_round_down = blocks_round_down
        self.normalize_offsets = normalize_offsets
        self.offsets_initializer = offsets_initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        nregions = _mex_dims_helper(input_shape[1:], self.num_instances, blocks=self.blocks, padding=self.padding,
                                    strides=self.strides, use_unshared_regions=self.use_unshared_regions,
                                    shared_offset_region=self.shared_offset_region,
                                    unshared_offset_region=self.unshared_offset_region)

        self.offsets = self.add_weight(name='offsets', shape=(nregions, self.num_instances) + tuple(self.blocks),
                                       trainable=True,
                                       initializer=self.offsets_initializer)

        if self.normalize_offsets:
            with tf.name_scope('NormalizeOffsets'):
                flattened = tf.reshape(self.offsets, [nregions, -1])
                norm_terms = tf.reduce_logsumexp(flattened, axis=1, keep_dims=True)
                flattened_normed = flattened / norm_terms # Broadcast
                self.offsets = tf.reshape(flattened_normed, tf.shape(self.offsets))

        super(Mex, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.op = _mex(x, self.offsets, self.num_instances, blocks=self.blocks, padding=self.padding,
                       strides=self.strides, use_unshared_regions=self.use_unshared_regions,
                       shared_offset_region=self.shared_offset_region,
                       unshared_offset_region=self.unshared_offset_region)
        return self.op

    def compute_output_shape(self, input_shape):
        return tuple(self.op.get_shape().as_list())


def perform_unsupervised_init(model, kind, layers=[], data=None, batch_size=None):
    if not layers:
        layers = [l for l in model.layers if isinstance(l, Similarity)]

    if isinstance(data, np.ndarray):
        if batch_size is None:
            raise ValueError('data is nparray, so batch size must be given')

        def data_gen():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size,...]
                yield batch
    else:
        data_gen = data

    input_tensor = model.input
    session = K.get_session()

    for layer in layers:
        u_init, u_op = _similarity_unsupervised_init(kind, layer.op, layer.weights[1], layer.weights[0])
        _initialize_variables()
        print('running unsupervised initialization for layer {}'.format(layer.name))
        for idx, batch in enumerate(data_gen()):
            fd = {input_tensor: batch}
            if idx == 0:
                session.run(u_init, feed_dict=fd)
            session.run(u_op, feed_dict=fd)

