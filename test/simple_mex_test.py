import sys
sys.path.append(r'/home/elhanani/study/huji-deep/install')
import simnets.keras as sk
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import add, Activation, RepeatVector, Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Reshape, Conv2D, AveragePooling2D, BatchNormalization, Lambda, ZeroPadding2D
from keras import backend as K
import numpy as np
np.random.seed(1)
with tf.device('/gpu:0'):
    data = np.arange(0,1, 0.001, dtype=np.float32).reshape(-1,1,1,1)
    y = (1 + data).reshape(-1,1)
    batch_size = 100

    input_shape = (1, 1, 1)


    tf.set_random_seed(1)
    a = Input(shape=(1, 1, 1))

    b = sk.Mex(3,
               blocks=[1, 1, 1], strides=[1, 1, 1],
               softmax_mode=False, normalize_offsets=False,
               use_unshared_regions=False, shared_offset_region=[-1])
    bb = b
    b = b(a)
    b = GlobalAveragePooling2D()(b)
    b = Lambda(lambda x: tf.reduce_mean(x, 1, keep_dims=True))(b)
    model = Model(inputs=[a], outputs=[b])

    print(model.summary())
    #import sys
    #sys.exit(0)

    cb = TensorBoard(histogram_freq=1, log_dir='/home/elhanani/study/huji-deep/install/logs',
                     write_graph=True, write_images=False, write_grads=True)
    #cb.validation_data = x_train
    #cb.set_model(model)
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.nadam(lr=3e-2, epsilon=1e-16))
    sess = K.get_session()

    model.fit(data, y,
             batch_size=batch_size,
             epochs=50,
             verbose=1,
             validation_data=(data, y), callbacks=[cb])
    with sess.as_default():
        fd = {
            'input_1:0': data[:1],
            'lambda_1_sample_weights:0': np.ones((1,), dtype=np.float32),
            'lambda_1_target:0': y[:1]

        }
        x, y, sample_weights = model._standardize_user_data(
            data, y,
            sample_weight=None,
            check_batch_axis=False,
            batch_size=32)
        computed, numeric = tf.test.compute_gradient(bb.weights[0], bb.weights[0].get_shape().as_list(),
                                                     model.total_loss, [1], delta=1e-3,
                                                     extra_feed_dict=fd)
        print(np.abs(computed - numeric).mean())
        np.save('/home/elhanani/tmp/computed.npy', computed)
        np.save('/home/elhanani/tmp/numeric.npy', numeric)
    score = model.evaluate(data, y, verbose=0)
    print('Test loss:', score)
    #print('Test accuracy:', score[1])
    print(model.predict(np.array([1,2,3,4,5,5.5]).reshape(-1,1,1,1)))
