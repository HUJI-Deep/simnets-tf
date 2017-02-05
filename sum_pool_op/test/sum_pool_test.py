import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) > 1 and sys.argv[1].endswith('so'):
    sum_op_module = tf.load_op_library(sys.argv[1])
else:
    import os
    so = os.path.abspath(os.path.join(os.path.split(__file__)[0], 'sum_pool_op.so'))
    sum_op_module = tf.load_op_library(so)
sum_pool = sum_op_module.sum_pool

@tf.RegisterGradient("SumPool")
def _sum_pool_grad(op, grad):
    inp = op.inputs[0]
    padding = op.get_attr('padding')
    strides = op.get_attr('strides')
    ksize = op.get_attr('ksize')
    grad_op = sum_op_module.sum_pool_grad(inp, grad, padding=padding, ksize=ksize, strides=strides)
    return grad_op

class SumPoolTest(tf.test.TestCase):
    def testAllOnesEvenValid(self):
        with self.test_session():
            x = tf.ones((4, 6, 8, 10), tf.float32)
            res = sum_pool(x, padding='VALID', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            self.assertAllEqual(res.eval(), 4 * np.ones((4, 3, 4, 10)))

    def testAllOnesEvenValidGrad(self):
        with self.test_session():
            x = tf.ones((4, 7, 7, 10), tf.int32)
            res = sum_pool(x, padding='VALID', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            res_grad = sum_op_module.sum_pool_grad(x, res, padding='VALID', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            self.assertAllEqual(res.eval(), 4 * np.ones((4, 3, 3, 10)))

    def testAllOnesOddValid(self):
        with self.test_session():
            x = tf.ones((4, 5, 7, 10), tf.float32)
            res = sum_pool(x, padding='VALID', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            self.assertAllEqual(res.eval(), 4 * np.ones((4, 2, 3, 10), np.float32))

    def testBigStrideSameEven(self):
        with self.test_session():
            x = np.zeros((4, 6, 6, 1), np.float32)
            x[:, 2, 2, :] = 1
            x = tf.constant(x)
            res = sum_pool(x, padding='SAME', ksize=[1, 2, 2, 1], strides=[1, 6, 6, 1])
            self.assertAllEqual(res.eval(), np.ones((4, 1, 1, 1), np.float32))

    def testSame(self):
        with self.test_session():
            x = tf.ones((1, 4, 6, 1), tf.float32)
            res = sum_pool(x, padding='SAME', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            r = res.eval()
            self.assertAllEqual(r, 4 * np.ones((1, 2, 3, 1), np.float32))

    def testHandCraftedSame1(self):
        a = np.array([[0, 1, 2, 1, 3, 1, 1],
                      [1, 1, 2, 4, 3, 1, 1],
                      [0, 1, 2, 1, 1, 1, 1],
                      [2, 1, 1, 1, 3, 1, 1]], np.float32)

        b = np.array([[3, 9, 8, 2],
                      [4, 5, 6, 2]], np.float32)

        a = a[None, ..., None]
        b = b[None, ..., None]
        with self.test_session():
            tfa = tf.constant(a, tf.float32)
            res = sum_pool(tfa, padding='SAME', ksize=[1,2,2,1], strides=[1,2,2,1])
            self.assertAllEqual(res.eval(), b)

    def testHandCraftedValid1(self):
        a = np.array([[0, 1, 2, 1, 3, 1, 1],
                      [1, 1, 2, 4, 3, 1, 1],
                      [0, 1, 2, 1, 1, 1, 1],
                      [2, 1, 1, 1, 3, 1, 1]], np.float32)

        b = np.array([[3, 9, 8],
                      [4, 5, 6]], np.float32)

        a = a[None, ..., None]
        b = b[None, ..., None]
        with self.test_session():
            tfa = tf.constant(a, tf.float32)
            res = sum_pool(tfa, padding='VALID', ksize=[1,2,2,1], strides=[1,2,2,1])
            self.assertAllEqual(res.eval(), b)

    def testGradientSame(self):
        x_shape = (3,8,7,2)
        y_shape = (1,)
        x_np = 6*np.random.normal(size=x_shape).astype(np.float32)
        with self.test_session():
            x = tf.constant(x_np)
            y = tf.reduce_sum(sum_pool(x, padding='SAME', ksize=[1,2,3,1], strides=[1,3,2,1]))
            symbolic, numeric = tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=x_np, delta=1e-1)
            self.assertAllClose(symbolic, numeric, atol=1e-2, rtol=1e-2)

    def testGradientValid(self):
        x_shape = (3,8,7,2)
        y_shape = (1,)
        x_np = 6*np.random.normal(size=x_shape).astype(np.float32)
        with self.test_session():
            x = tf.constant(x_np)
            y = tf.reduce_sum(sum_pool(x, padding='VALID', ksize=[1,2,3,1], strides=[1,3,2,1]))
            symbolic, numeric = tf.test.compute_gradient(x, x_shape, y, y_shape, x_init_value=x_np, delta=1e-1)
            self.assertAllClose(symbolic, numeric, atol=1e-2, rtol=1e-2)



if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].endswith('so'):
        sys.argv = sys.argv[0:1] + sys.argv[2:]
    tf.test.main()
