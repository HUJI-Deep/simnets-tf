# simnets-tf

SimNets is a generalization of Convolutional Neural Networks that was first proposed by Cohen et al. (at [NIPS 2014 DL Workshop](https://arxiv.org/abs/1410.078) and [CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Cohen_Deep_SimNets_CVPR_2016_paper.html)), in whish they showed it to be well-suited for classification under limited computional resources. It was later discovered that under some settings SimNets realize [Convolutional Arithmetic Circuits](http://www.jmlr.org/proceedings/papers/v49/cohen16.html), where all computation is done in log-space instead of linear-space. We have previously released an implementation of [SimNets in Caffe](https://github.com/HUJI-Deep/caffe-simnets), and now we release our implementaiton in Tensorflow, with wrapper layers for Keras. You can read the documentation at https://huji-deep.github.io/simnets-tf/.

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
 git clone --recursive https://github.com/HUJI-Deep/simnets-tf.git
 cd simnets-tf
 mkdir build
 cd build
 cmake .. -DCMAKE_BUILD_TYPE=Release -DSIMNETS_PYTHON_VERSION=2.7 -DCMAKE_INSTALL_PREFIX=install
 make -j simnet_ops
 ```
 
 Python 3.5:<BR/>
  ```
  git clone --recursive https://github.com/HUJI-Deep/simnets-tf.git
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
                  ksize=[2, 2], strides=[2, 2], similarity_function='L2',
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

```

#### Low level
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from simnets import similarity
from simnets.unsupervised import similarity_unsupervised_init
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
xr = tf.reshape(x, [-1, 1, 28, 28])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

shape = [10, 1, 28, 28]

templates = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
weights_var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
weights = tf.abs(weights_var)

sim = similarity(xr, templates, weights, similarity_function='L2', ksize=[28,28], strides=[28,28], padding=[0,0])
y = tf.reshape(sim, [-1, 10])

kmo_init, kmo = similarity_unsupervised_init('kmeans', sim, templates, weights_var)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())


for idx in range(300):
    batch = mnist.train.next_batch(100)
    if idx == 0:
        kmo_init.run(feed_dict={x: batch[0], y_: batch[1]})
    kmo.run(feed_dict={x: batch[0], y_: batch[1]})
    if (idx + 1) % 100 == 0:
        print('kmeans', idx+1, '/', 1000)

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

templates_np = tf.get_default_session().run(templates)
plt.figure(1)
for i in range(10):
    plt.subplot(4,3, i+1)
    plt.imshow(normalize(templates_np[i,0,...]))
plt.show()

for idx in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if (idx + 1) % 100 == 0:
        print(idx+1, '/', 1000)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

```
 
