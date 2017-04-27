import tensorflow as tf
from keras.engine.topology import Layer
from .ops import similarity as _similarity
from .ops.mex import _mex_dims_helper, mex as _mex

class Similarity(Layer):

    def __init__(self, num_instances, similarity_function='L2', strides=[1,1], ksize=[3,3], padding='SAME',
                 normalization_term=False, normalization_term_fudge=0.001, ignore_nan_input=False,
                 out_of_bounds_value=0, **kwargs):
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


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sweights = self.add_weight(shape=(self.num_instances, input_shape[1], self.ksize[0], self.ksize[1]),
                                        trainable=True,
                                        initializer='uniform')
        self.templates = self.add_weight(shape=(self.num_instances, input_shape[1], self.ksize[0], self.ksize[1]),
                                         initializer='uniform',
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

class Mex(Layer):

    def __init__(self, num_instances, padding=[0, 0, 0], strides=[1, 1, 1], blocks=[1, 1, 1],
                 epsilon=1.0, use_unshared_regions=True, shared_offset_region=[-1],
                 unshared_offset_region=[-1], softmax_mode=False, blocks_out_of_bounds_value=0.0,
                 blocks_round_down=True, **kwargs):
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


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        nregions = _mex_dims_helper(input_shape[1:], self.num_instances, blocks=self.blocks, padding=self.padding,
                                    strides=self.strides, use_unshared_regions=self.use_unshared_regions,
                                    shared_offset_region=self.shared_offset_region,
                                    unshared_offset_region=self.unshared_offset_region)

        self.offsets = self.add_weight(shape=(nregions, self.num_instances, *self.blocks),
                                        trainable=True,
                                        initializer='glorot_uniform')

        super(Mex, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.op = _mex(x, self.offsets, self.num_instances, blocks=self.blocks, padding=self.padding,
                       strides=self.strides, use_unshared_regions=self.use_unshared_regions,
                       shared_offset_region=self.shared_offset_region,
                       unshared_offset_region=self.unshared_offset_region)
        return self.op

    def compute_output_shape(self, input_shape):
        return tuple(self.op.get_shape().as_list())
