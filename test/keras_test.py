import sys
sys.path.append(r'/home/elhanani/study/huji-deep/install')
import tensorflow as tf
print(tf.__version__)
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

with tf.device('/cpu:0'):
    kmo_init, kmo = similarity_unsupervised_init('gmm', sim, templates, weights_var)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

batch = mnist.train.next_batch(100)
sess.run(tf.global_variables_initializer(), feed_dict={x: batch[0], y_: batch[1]})


for idx in range(300):
    batch = mnist.train.next_batch(1000)
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