import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

def kmeans_unsupervised_init(sim_op, templates_var, weights_var):
    if isinstance(sim_op, tf.Tensor):
        sim_op = sim_op.op
    if not sim_op.type == 'Similarity':
        raise ValueError('kmeans_unsupervised_init needs a similarity op, got %s instead' % sim_op.type)
    assert(isinstance(sim_op, tf.Operation))
    name = sim_op.name + '_kmeans_init'
    with tf.name_scope(name):
        input_tensor = sim_op.inputs[0]
        templates_tensor = sim_op.inputs[1]
        weights_tensor = sim_op.inputs[2]
        ninstances = templates_tensor.get_shape().as_list()[0]

        strides = sim_op.get_attr('strides')
        blocks = sim_op.get_attr('ksize')
        strides = [1, strides[0], strides[1], 1]
        blocks = [1, blocks[0], blocks[1], 1]
        patches = tf.extract_image_patches(tf.transpose(input_tensor, (0, 2, 3, 1)), strides=strides,
                                           ksizes=blocks, rates=[1, 1, 1, 1], padding='VALID')
        _, _, _, ppatch = patches.get_shape().as_list()
        patches = tf.reshape(patches, [-1, ppatch])
        kmeans = KMeans(patches, ninstances, use_mini_batch=True, initial_clusters='kmeans_plus_plus')
        _, _, _, _, init_op, training_op = kmeans.training_graph()
        clusters_var = [v for v in tf.global_variables() if v.name == name + '/' + 'clusters:0'][0]
        clusters = clusters_var.op.outputs[0]

        channels, block_rows, block_cols = templates_tensor.get_shape().as_list()[1:]
        reshaped_clusters = tf.reshape(clusters, (ninstances, block_rows, block_cols, channels))
        transposed_clusters = tf.transpose(reshaped_clusters, [0, 3, 1, 2])
        with tf.control_dependencies([training_op]):
            assign1 = tf.assign(templates_var, transposed_clusters)
            assign2 = tf.assign(weights_var, tf.ones_like(transposed_clusters))
        return init_op, tf.group(assign1, assign2, name='kmeans_init_assign')

