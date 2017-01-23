import tensorflow as tf
with tf.device('/cpu:0'):
    m = tf.load_op_library('../cmake-build-release/sum_pool_op.so')
    sess = tf.Session()
    o = tf.ones(shape=(10,12,14,16), dtype=tf.float32)
    #o = tf.placeholder(tf.float32, (10, 12, 14, 1))
    myop = m.sum_pool(o, 'VALID', ksize=[1,2,2,1], strides=[1,1,1,1])
    print('Before eval', myop.get_shape())
    r = myop.eval(session=sess)
    print(r[0])
    print(r.shape)
