# simnets-tf
## SimNets implementation in TensorFlow

### Binary installation
Binary installation requires a cuda toolkit installation >= 7.5. <BR/>
Download the .whl file from the GitHub release tab, then type:
```
python -m pip install <whl file>
```
all requirements should be installed automatically.

### Building from Source
Building from source requires:
1. A working c++ compiler with c++11 support (gcc >= 4.7)
2. Cuda toolkit installed (for nvcc)
3. CMake >= 3.0 (<code>apt install cmake</code>)
4. TensorFlow installed for the Python interpreter you intend to use

<B>Important:</B> The following command should run without error:
```
python -c 'import tensorflow as tf'
```
To build the project type the following commands:<BR/>
 Python 2.7:<BR/>
 ```
 git clone https://github.com/HUJI-Deep/simnets-tf.git
 cd simnets-tf
 mkdir build
 cd build
 cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMNETS_PYTHON_VERSION=2.7 -DCMAKE_INSTALL_PREFIX=install
 make -j simnet_ops
 ```
 
 Python 3.5:<BR/>
  ```
  git clone https://github.com/HUJI-Deep/simnets-tf.git
  cd simnets-tf
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMNETS_PYTHON_VERSION=3.5 -DCMAKE_INSTALL_PREFIX=install
  make -j simnet_ops
  ```
 To test the code you can now type:
 ```
 make test_simnet_ops
 ```
 This should run for about two minutes and return without any errors.<BR/>
 Now you can create a .whl file:
 ```
 make create_wheel
 ```
 
 Finally, to install the simnets-tf package type (remember to use the right interpreter):
 ```
 cd install/dist
 python -m pip install <whl file>
 ```
 The installation is successful if the following runs (again, remember to use the right interpreter):
 ```
 python -c 'import simnets'
 ```
 
 ### Usage example
 #### Keras
 ```python
import simnets.keras as sk
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
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
model.add(sk.Similarity(64, ksize=[1, 1], strides=[1, 1], similarity_function='L2', input_shape=input_shape))
model.add(sk.Mex(64, blocks=[64, 3, 3], strides=[64, 3, 3]))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

```

#### Low level
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from simnets import similarity

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
xr = tf.reshape(x, [-1, 1, 28, 28])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

shape = [10, 1, 28, 28]

templates = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
weights = tf.abs(tf.Variable(tf.truncated_normal(shape, stddev=0.1)))

sess.run(tf.global_variables_initializer())
y = similarity(xr, templates, weights, similarity_function='L2', ksize=[28,28], strides=[28,28], padding=[0,0])
y = tf.reshape(y, [-1, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for idx in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if (idx + 1) % 100 == 0:
        print(idx+1, '/', 1000)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
 
