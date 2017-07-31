import simnets.keras as sk
import keras
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import add, RepeatVector, Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Reshape, Conv2D, AveragePooling2D, BatchNormalization, Lambda, ZeroPadding2D
from keras import backend as K
import tensorflow as tf
import numpy as np

batch_size = 128
num_classes = 10
sim_kernel = 2
sim_channels = 32
mex_channels = sim_channels
epochs = 30
generative_loss_weight = 1e-2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


assert(K.image_data_format() == 'channels_first')
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.mean(), x_train.min(), x_train.max())
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5

print(x_train.mean(), x_train.min(), x_train.max())

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# def norm_layer(x):
#     norm = sk.Mex(1, blocks=[int(x.shape[-3]), 1, 1],
#         softmax_mode=True, normalize_offsets=False,
#         use_unshared_regions=False, shared_offset_region=[-1],
#         offsets_initializer='zeros',
#         trainable=False)(x)
#     tile = Reshape((int(x.shape[-2]) * int(x.shape[-1]),))(norm)
#     tile = RepeatVector(int(x.shape[-3]))(tile)
#     tile = Reshape(x._keras_shape[-3:])(tile)
#     tile = Lambda(lambda x: -x)(tile)
#     normalized = add([x, tile])
#     return normalized, norm

def sum_pooling_layer(x, pool_size=(2, 2)):
    # Average pooling doesn't take padding area into consideration when using padding='same'
    # If we wish to add padding, we must do so externally with ZeroPadding2D so we would
    # be able to compute the sum pooling layer from the average pooling layer.
    x = AveragePooling2D(pool_size=pool_size, padding='valid')(x)
    x = Lambda(lambda x: x * pool_size[0] * pool_size[1])(x)
    return x


def fixed_dirichlet_init(shape, dtype=None):
    if dtype is None:
        dtype = K.floatx()
    num_regions, num_instances, block_c, block_h, block_w = shape
    k = block_c * block_h * block_w
    # when given s as a size argument dirichlet function return an array with shape s + [k]
    # then we reshape the output to be of the same shape as the variable
    init_np = np.random.dirichlet([1] * k, size=(num_regions, num_instances)).astype(dtype)
    init_np = np.log(init_np)
    init_np = init_np.reshape(shape)
    return tf.constant(init_np)

a = Input(shape=(1, img_rows, img_cols))
b = sk.Similarity(sim_channels,
    blocks=[2, 2], strides=[2, 2], similarity_function='L2',
    normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True,
    normalization_term_fudge=1e-4, weights_initializer=keras.initializers.Constant(value=100))(a)
i = 0
# last_norm=None
while b.shape[-2:] != (1, 1):
    mex_channels *= 2
    unshared = 2 if i < 1 else int(b.shape[-2])
    #b, b_norm = norm_layer(b)
    b = sk.Mex(mex_channels,
        blocks=[int(b.shape[-3]), 1, 1], strides=[int(b.shape[-3]), 1, 1],
        softmax_mode=True, normalize_offsets=True,
        use_unshared_regions=True, unshared_offset_region=[unshared],
        offsets_initializer=fixed_dirichlet_init)(b)
    b = sum_pooling_layer(b, pool_size=(2, 2))
    # b_norm = sum_pooling_layer(b_norm, pool_size=(2, 2))
    # if last_norm is None:
    #     last_norm = b_norm
    # else:
    #     last_norm = sum_pooling_layer(last_norm, pool_size=(2, 2))
    #     last_norm = add([last_norm, b_norm])
    i += 1
#b, b_norm = norm_layer(b)
b = sk.Mex(num_classes,
    blocks=[mex_channels, 1, 1], strides=[mex_channels, 1, 1],
    softmax_mode=True, normalize_offsets=True,
    use_unshared_regions=True, shared_offset_region=[1],
    offsets_initializer=fixed_dirichlet_init)(b)
b = Flatten()(b)
model = Model(inputs=[a], outputs=[b])

# print(model.summary())

def generative_classifier_loss(y_true, y_pred):
    discriminative_loss = K.categorical_crossentropy(y_pred, y_true, True)
    generative_loss = -tf.reduce_mean(tf.reduce_logsumexp(y_pred, axis=1) + np.log(1.0 / num_classes)) # add 1/K factor!
    return discriminative_loss + generative_loss_weight * generative_loss

model.compile(loss=generative_classifier_loss,
             optimizer=keras.optimizers.nadam(lr=3e-2, epsilon=1e-16),
             metrics=['accuracy'])

# sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)

model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test),
         callbacks=[TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)])

model.save_weights('mnist_ght_model_weights.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Clean Test Dataset | loss: ' , score[0], ' accuracy:', score[1])

iid_p = 0.9
x_test_missing_iid = x_test.copy()
x_test_missing_iid[np.random.random_sample(size=x_test.shape) < iid_p] = np.nan
score_missing = model.evaluate(x_test_missing_iid, y_test, verbose=0)
print('Test Dataset with %d%% Missing iid | ' % (iid_p * 100), 'loss: ' , score_missing[0], ' accuracy:', score_missing[1])
