import sys
sys.path.append(r'/home/elhanani/study/huji-deep/install')
import simnets.keras as sk
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras import backend as K

batch_size = 32
num_classes = 10
epochs = 2

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
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(9, kernel_size=[3, 3], strides=[1, 1], input_shape=input_shape))
model.add(sk.Similarity(64, ksize=[1, 1], strides=[1, 1], similarity_function='L2'))
model.add(sk.Mex(64, blocks=[64, 3, 3], strides=[64, 3, 3]))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
import tensorflow as tf

sk.perform_unsupervised_init(model, kind='gmm', data=x_train, batch_size=batch_size)
clusters = tf.get_default_graph().get_tensor_by_name('similarity_1/weights:0')
clusters = session = K.get_session().run(clusters)
#print(clusters.mean(axis=(1,2,3)))
import sys

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])