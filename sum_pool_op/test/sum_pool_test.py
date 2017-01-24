import tensorflow as tf
import numpy as np
import sys

sum_op_module = tf.load_op_library(sys.argv[1])
sum_pool = sum_op_module.sum_pool

class SumPoolTest(tf.test.TestCase):

    def testAllOnesEvenValid(self):
        with self.test_session():
            x = tf.ones((4,6,8,10), tf.float32)
            res = sum_pool(x, 'VALID', [1,2,2,1], [1,2,2,1])
            self.assertAllEqual(res.eval(), 4*np.ones((4,3,4,10)))

    def testAllOnesOddValid(self):
        with self.test_session():
            x = tf.ones((4,5,7,10), tf.float32)
            res = sum_pool(x, 'VALID', [1,2,2,1], [1,2,2,1])
            self.assertAllEqual(res.eval(), 4*np.ones((4,2,3,10), np.float32))

    def testBigStrideSameEven(self):
        with self.test_session():
            x = np.zeros((4,6,6,2), np.float32)
            x[:,2,2,:] = 1
            x = tf.constant(x)
            res = sum_pool(x, 'SAME', ksize=[1,2,2,1], strides=[1,6,6,1])
            self.assertAllEqual(res.eval(), np.ones((4,1,1,2), np.float32))

    def testSame(self):
        with self.test_session():
            x = tf.ones((1,4,6,1), tf.float32)
            res = sum_pool(x, 'SAME', ksize=[1,2,2,1], strides=[1,2,2,1])
            r = res.eval()
            self.assertAllEqual(r, 4*np.ones((1,2,3,1), np.float32))


if __name__ == '__main__':
    sys.argv = sys.argv[0:1] + sys.argv[2:]
    tf.test.main()