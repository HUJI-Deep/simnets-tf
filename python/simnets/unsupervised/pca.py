import tensorflow as tf

'''
class RS(object):
    ...:     def __init__(self):
        ...:         self.n = 0
    ...:         self.new_m = 0
    ...:         self.new_s = 0
    ...:     def add(self, vals):
        ...:         if self.n == 0:
        ...:             self.n += len(vals)
    ...:             self.new_m = np.mean(vals)
    ...:         else:
    ...:             old_m = self.new_m
    ...:             self.n += len(vals)
    ...:             self.new_m = old_m + (np.mean(vals) - old_m) / self.n
    ...:             self.new_s = self.new_s + (np.mean(vals) - old_m)*(np.mean(
        ...: vals) - self.new_m)
'''

# gsm cmd line srun --mem=3G -c2 --gres=gpu --time=48:0:00 --pty csh
# module load cuda cudnn
class RunningAverage(object):

    def __init__(self, shape, dtype):
        self.n = tf.get_variable('pca_n', shape=[], dtype=tf.int64, initializer=tf.zeros_initializer)
        self.m = tf.get_variable('pca_m', shape=shape, dtype=dtype, initializer=tf.zeros_initializer)
        self.s = tf.get_variable('pca_s', shape=shape, dtype=dtype, initializer=tf.zeros_initializer)

    def add(self, values):
        mean_values = tf.reduce_mean(values, axis=0)

        # initial case
        len_values = tf.cast(tf.shape(values)[0], tf.int64)
        init_n = tf.assign(self.n, len_values)
        init_m = tf.assign(self.m, mean_values)
        first_time_op = tf.group(init_n, init_m)

        # normal case
        updated_n = self.n + len_values
        updated_m = self.m + (mean_values - self.m) / tf.cast(updated_n, tf.float32)
        updated_s = self.s + (mean_values - self.m) * (mean_values - updated_m)
        with tf.control_dependencies([updated_n, updated_m, updated_s]):
            update_n = tf.assign(self.n, updated_n)
            update_m = tf.assign(self.m, updated_m)
            update_s = tf.assign(self.s, updated_s)
        every_time_op = tf.group(update_n, update_m, update_s)
        return tf.cond(tf.equal(self.n, 0), lambda: first_time_op, lambda: every_time_op)

    def value(self):
        return self.m


if __name__ == '__main__':
    ra = RunningAverage([3], tf.float32)
    vals = tf.placeholder(tf.float32, [500,3])
    add_op = ra.add(vals)
    m = ra.value()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    import numpy as np
    for i in range(300):
        sess.run(add_op, feed_dict={vals: np.random.normal(3.0, size=[500,3])})
    print(sess.run(m))