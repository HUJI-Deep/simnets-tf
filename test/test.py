import tensorflow as tf
import numpy as np

so = tf.load_op_library('./simnet_ops.so')
similarity = so.similarity
similarity_ref = so.similarity_ref

class SimilarityTests(tf.test.TestCase):

    def testSanity(self):
        with self.test_session():
            #images = np.random.normal(size=(1,1,7,10)).astype(np.float32)
            images = np.ones((1,1,1,4), np.float32)
            images = tf.constant(images)

            #weights = np.absolute(np.random.normal(size=(1,1,3,3)).astype(np.float32))
            weights = np.ones((1,1,3,3), np.float32)
            weights = tf.constant(weights)

            #templates = np.random.normal(size=(1,1,3,3)).astype(np.float32)
            templates = np.zeros((1,1,3,3), np.float32)
            templates = tf.constant(templates)

            sim = similarity(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            sim_ref = similarity_ref(images, templates, weights, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
            s = sim.eval()
            sr = sim_ref.eval()
            self.assertAllClose(s, sr, 1e-4)


if __name__ == '__main__':
    tf.test.main()
