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
    batch_size = 100
    num_classes = 10
    sim_kernel = 2
    sim_channels = 32
    mex_channels = sim_channels
    epochs = 1
    weights = np.load('/home/elhanani/notebooks/weights.npy')[()]
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train) = (x_train[:1000], y_train[:1000])
    (x_test, y_test) = (x_test[:1000], y_test[:1000])
    (x_test, y_test) = (x_train, y_train)

    assert(K.image_data_format() == 'channels_first')
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.mean(), x_train.min(), x_train.max())
    x_train = 0*x_train / 255.0 - 0.5
    x_test = 0*x_test / 255.0 - 0.5

    print(x_train.mean(), x_train.min(), x_train.max())

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def norm_layer(x):
        norm = sk.Mex(1, blocks=[int(x.shape[-3]), 1, 1],
            softmax_mode=True, normalize_offsets=False,
            use_unshared_regions=False, shared_offset_region=[-1],
            offsets_initializer='zeros',
            trainable=False)(x)
        tile = Reshape((int(x.shape[-2]) * int(x.shape[-1]),))(norm)
        tile = RepeatVector(int(x.shape[-3]))(tile)
        tile = Reshape(x._keras_shape[-3:])(tile)
        tile = Lambda(lambda x: -x)(tile)
        normalized = add([x, tile])
        return normalized, norm

    def sum_pooling_layer(x, pool_size=(2, 2)):
        x = AveragePooling2D(pool_size=pool_size)(x)
        x = Lambda(lambda x: x * pool_size[0] * pool_size[1])(x)
        return x
    tf.set_random_seed(1)
    a = Input(shape=(1, img_rows, img_cols))
    b = sk.Similarity(sim_channels,
                     blocks=[2, 2], strides=[2, 2], similarity_function='L2',
                     normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True,
                     normalization_term_fudge=1e-4, templates_initializer=tf.constant_initializer(weights['sim_templates']),
                      weights_initializer=tf.constant_initializer(weights['sim_weights']))(a)
    # b = sk.Similarity(sim_channels,
    #     blocks=[2, 2], strides=[2, 2], similarity_function='L2',
    #     normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True)(a)#,
    #                  # templates_initializer=tf.constant_initializer(weights['sim_templates']),
    #                  # weights_initializer=tf.constant_initializer(weights['sim_weights']))(a)
    #b = a
    i = 0
    # last_norm=None
    bb = None
    while b.shape[-2:] != (1, 1):
        with tf.name_scope('block'):
            i += 1
            mex_channels *= 2
            unshared = 2  # if i < 1 else -1
            #b, b_norm = norm_layer(b)
            l = sk.Mex(mex_channels,
                blocks=[int(b.shape[-3]), 1, 1], strides=[int(b.shape[-3]), 1, 1],
                softmax_mode=True, normalize_offsets=True, blocks_out_of_bounds_value=np.nan,
                use_unshared_regions=True, unshared_offset_region=[unshared],
                       offsets_initializer=tf.constant_initializer(weights['mex' + str(i)]))
            print('Shape:', b.shape[-3])
            if bb is None:
                bb = l
                with open('/home/elhanani/tmp/mex_dict.txt', 'w') as f:
                    d = dict(ninstances=mex_channels,
                           blocks=[int(b.shape[-3]), 1, 1], strides=[int(b.shape[-3]), 1, 1],
                           softmax_mode=True, normalize_offsets=True, blocks_out_of_bounds_value=np.nan,
                           use_unshared_regions=True, unshared_offset_region=[unshared], offsets_initializer=tf.zeros_initializer())
                    import pprint
                    pprint.pprint(str(d), stream=f)

            b = l(b)
            b = sum_pooling_layer(b, pool_size=(2, 2))
            # b_norm = sum_pooling_layer(b_norm, pool_size=(2, 2))
            # if last_norm is None:
            #     last_norm = b_norm
            # else:
            #     last_norm = sum_pooling_layer(last_norm, pool_size=(2, 2))
            #     last_norm = add([last_norm, b_norm])
    #b, b_norm = norm_layer(b)
    b = sk.Mex(num_classes,
        blocks=[mex_channels, 1, 1], strides=[mex_channels, 1, 1],
        softmax_mode=True, normalize_offsets=True,
        use_unshared_regions=False, shared_offset_region=[-1],
               offsets_initializer=tf.constant_initializer(weights['mex5']))(b)
    b = Flatten()(b)
    #b = Activation('softmax')(b)
    model = Model(inputs=[a], outputs=[b])

    print(model.summary())

    def softmax_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true, True)

    cb = TensorBoard(histogram_freq=1, log_dir='/home/elhanani/study/huji-deep/install/logs',
                     write_graph=True, write_images=False, write_grads=True)
    #cb.validation_data = x_train
    #cb.set_model(model)
    model.compile(loss=softmax_loss,
                 optimizer=keras.optimizers.sgd(),#keras.optimizers.nadam(lr=3e-2, epsilon=1e-16),
                 metrics=['accuracy'])
    sess = K.get_session()

    # sk.perform_unsupervised_init(model, 'kmeans', layers=None, data=x_train, batch_size=100)
    #ys = model.predict(x_test)
    #print(ys)
    #print(y_test)
    #print('Accuracy:', np.mean(np.argmax(ys, axis=1) == np.argmax(y_test, axis=1)))
    #print(model.evaluate(x_train, y_train, verbose=1))
    model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             validation_data=(x_test, y_test), callbacks=[cb])
    # with sess.as_default():
    #     fd = {
    #         'input_1:0': x_train[:1],
    #         'flatten_1_sample_weights:0': np.ones((1,), dtype=np.float32),
    #         'flatten_1_target:0': y_test[:1]
    #
    #     }
    #     x, y, sample_weights = model._standardize_user_data(
    #         x_test, y_test,
    #         sample_weight=None,
    #         check_batch_axis=False,
    #         batch_size=32)
    #     computed, numeric = tf.test.compute_gradient(bb.weights[0], bb.weights[0].get_shape().as_list(),
    #                                                  model.total_loss, [1], delta=1e-3,
    #                                                  extra_feed_dict=fd)
    #     print(np.abs(computed - numeric).mean())
    #     np.save('/home/elhanani/tmp/computed.npy', computed)
    #     np.save('/home/elhanani/tmp/numeric.npy', numeric)
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
