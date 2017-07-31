import sys
sys.path.append(r'/home/elhanani/study/huji-deep/install')
import simnets.keras as sk
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, AveragePooling2D, Lambda
from keras import backend as K
import numpy as np

batch_size = 64
num_classes = 10
sim_kernel = 2
sim_channels = 32
mex_channels = sim_channels
epochs = 3

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def sum_pooling_layer(x, pool_size):
    x = AveragePooling2D(pool_size=pool_size, padding='valid')(x)
    x = Lambda(lambda x: x * pool_size[0] * pool_size[1])(x)
    return x


a = Input(shape=(1, img_rows, img_cols))
b = sk.Similarity(sim_channels,
                  blocks=[2, 2], strides=[2, 2], similarity_function='L2',
                  normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True)(a)
while b.shape[-2:] != (1, 1):
    mex_channels *= 2
    b = sk.Mex(mex_channels,
               blocks=[int(b.shape[-3]), 1, 1], strides=[int(b.shape[-3]), 1, 1],
               softmax_mode=True, normalize_offsets=True,
               use_unshared_regions=True, unshared_offset_region=[2])(b)
    b = sum_pooling_layer(b, pool_size=(2, 2))

b = sk.Mex(num_classes,
           blocks=[mex_channels, 1, 1], strides=[mex_channels, 1, 1],
           softmax_mode=True, normalize_offsets=True,
           use_unshared_regions=True, shared_offset_region=[1])(b)
b = Flatten()(b)
model = Model(inputs=[a], outputs=[b])

print(model.summary())

def softmax_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true, True)

model.compile(loss=softmax_loss,
              optimizer=keras.optimizers.nadam(lr=1e-2, epsilon=1e-6),
              metrics=['accuracy'])

sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])