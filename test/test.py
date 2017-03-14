import tensorflow as tf
import numpy as np

so = tf.load_op_library('./simnet_ops.so')
similarity = so.similarity
similarity_ref = so.similarity_ref

class SimilarityTests(tf.test.TestCase):

    def testSanity(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,3,800,800), np.float32)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,3,3,3), np.float32)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,3,3,3), np.float32)
            templates = tf.constant(templates)

            with tf.device('/cpu:0'):
                sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,1,1], padding='SAME', normalization_term=True)
            sim_ref = similarity_ref(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,1,1], padding='SAME', normalization_term=True)
            s = sim.eval()
            sr = sim_ref.eval()
            self.assertAllClose(s, sr, 1e-4)


if __name__ == '__main__':
    tf.test.main()
