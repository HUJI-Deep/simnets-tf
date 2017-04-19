from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine.topology import Layer
from .ops import similarity
import numpy as np

class Similarity(Layer):

    def __init__(self, output_dim, **kwargs):
        print('init')
        self.output_dim = output_dim
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        print('build')
        print(input_shape)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Similarity, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print('call')
        return similarity(x, )
        #return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        print('compute_output_shape')
        return (input_shape[0], self.output_dim)