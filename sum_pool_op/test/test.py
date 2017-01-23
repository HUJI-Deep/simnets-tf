import tensorflow as tf
m = tf.load_op_library('../cmake-build-debug/sum_pool_op.so')
sess = tf.Session()
#o = tf.ones(shape=(10,12,14,16), dtype=tf.float32)
o = tf.placeholder(tf.float32, (10, 12, 14, 16))
myop = m.sum_pool(o, 'SAME', [1,2,2,1],[1,2,2,1])
print('Before eval', myop.get_shape())
r = myop.eval(session=sess)
print(r.shape)
