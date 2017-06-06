import tensorflow as tf


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


def pca_unsupervised_init(conv_op, filters_var):
    if isinstance(conv_op, tf.Tensor):
        conv_op = conv_op.op
    if not conv_op.type == 'Conv2D':
        raise ValueError('pca_unsupervised_init needs a convolution op, got %s instead' % conv_op.type)
    assert(isinstance(conv_op, tf.Operation))
    name = conv_op.name + '_pca_init'
    with tf.name_scope(name):
        with tf.variable_scope(name):
            input_tensor = conv_op.inputs[0]
            filter_height, filter_width, in_channels, out_channels = filters_var.get_shape().as_list()
            single_filter_size = filter_height * filter_width * in_channels
            with tf.variable_scope('mu'):
                mu_manager = RunningAverage([single_filter_size], filters_var.dtype)
            with tf.variable_scope('sigma'):
                sigma_manager = RunningAverage([single_filter_size, single_filter_size], filters_var.dtype)

            ninstances = out_channels

            strides = conv_op.get_attr('strides')
            blocks = [1, filter_height, filter_width, 1]
            data_format = conv_op.get_attr('data_format')
            if data_format == 'NCHW':
                input_tensor = tf.transpose(input_tensor, (0, 2, 3, 1))
            patches = tf.extract_image_patches(input_tensor, strides=strides,
                                               ksizes=blocks, rates=[1, 1, 1, 1], padding='VALID')
            _, _, _, patch_size = patches.get_shape().as_list()
            patches = tf.reshape(patches, [-1, patch_size])
            mu_update = mu_manager.add(patches)
            sigma_update = sigma_manager.add(tf.matmul(tf.expand_dims(patches, 2), tf.expand_dims(patches, 1)))
            update_op = tf.group(mu_update, sigma_update)
            cov = sigma_manager.value() - tf.matmul(tf.expand_dims(mu_manager.value(), 1),
                                                    tf.expand_dims(mu_manager.value(), 0))
            s, u, v = tf.svd(cov)
            pca = v[:, :ninstances]
            pca_for_filters = tf.reshape(pca, filters_var.get_shape().as_list())
            assign_op = tf.assign(filters_var, pca_for_filters)
            return update_op, assign_op

if __name__ == '__main__':
    vals = tf.placeholder(tf.float32, [3,200,100,2])
    filters = tf.get_variable('filts', shape=[3, 3, 2, 4])
    conv = tf.nn.conv2d(vals, filters, strides=[1,2,2,1], padding='VALID')
    conv2 = tf.nn.conv2d(vals, filters, strides=[1,2,2,1], padding='VALID')
    update_op, assign_op = pca_unsupervised_init(conv, filters)
    update_op2, assign_op2 = pca_unsupervised_init(conv2, filters)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    import numpy as np
    for i in range(30):
        sess.run(update_op, feed_dict={vals: np.random.normal(3.0, size=[3,200,100,2])})
    sess.run(assign_op)