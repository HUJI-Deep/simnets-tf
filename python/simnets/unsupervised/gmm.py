import tensorflow as tf
from tensorflow.contrib.factorization import gmm as _gmm


def gmm_unsupervised_init(sim_op, templates_var, weights_var):
    """Initialize a similarity layer using gmm unsupervised learning

    Initializes the templates and weights using gmm
    The function returns two ops. The first is used to initialize the learning and the second should be run iteratively
     with all the data.

    Parameters
    ----------
    sim_op : tf.Operation | tf.Tensor
        the similarity operation (or the tensor which is the output of the similarity)
    templates_var : tf.Variable
        the templates variable for this similarity layer
    weights_var : tf.Variable
        the weights variable for this similarity layer

    Returns
    -------
    A tuple (init_op, update_op) where init_op must be executed by a session before using the update op
    and the update_op is the operation that performs the learning.
    """
    if isinstance(sim_op, tf.Tensor):
        sim_op = sim_op.op
    if not sim_op.type == 'Similarity':
        raise ValueError('kmeans_unsupervised_init needs a similarity op, got %s instead' % sim_op.type)
    assert(isinstance(sim_op, tf.Operation))
    name = sim_op.name + '_gmm_init'
    with tf.name_scope(name):
        input_tensor = sim_op.inputs[0]
        templates_tensor = sim_op.inputs[1]
        num_instances = templates_tensor.get_shape().as_list()[0]

        strides = sim_op.get_attr('strides')
        blocks = sim_op.get_attr('blocks')
        strides = [1, strides[0], strides[1], 1]
        blocks = [1, blocks[0], blocks[1], 1]
        patches = tf.extract_image_patches(tf.transpose(input_tensor, (0, 2, 3, 1)), strides=strides,
                                           blocks=blocks, rates=[1, 1, 1, 1], padding='VALID')
        _, _, _, patch_size = patches.get_shape().as_list()
        patches = tf.reshape(patches, [-1, patch_size])
        _, _, _, training_op = _gmm(inp=patches, initial_clusters='random',
                                    random_seed=33, num_clusters=num_instances, covariance_type='diag', params='mc')
        clusters_var = [v for v in tf.global_variables() if v.name == name + '/' + 'clusters:0'][0]
        clusters = clusters_var.op.outputs[0]
        clusters_covs_var = [v for v in tf.global_variables() if v.name == name + '/' + 'clusters_covs:0'][0]
        clusters_covs = clusters_covs_var.op.outputs[0]

        # this hacky code makes sure that the gmm code does not add a variable initializer that depends
        # on the input, which is usually a placeholder. without it, the global intializer must be run with
        # a feed dict, which dows work for keras, and is weird for other code
        non_gmm_vars = [v for v in tf.global_variables() if not v.name.startswith(name)]
        gmm_vars = [v for v in tf.global_variables() if v.name.startswith(name)]
        graph = tf.get_default_graph()
        graph.clear_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in non_gmm_vars:
            graph.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
        initializer = tf.group(*[v.initializer for v in gmm_vars])

        channels, block_rows, block_cols = templates_tensor.get_shape().as_list()[1:]
        reshaped_clusters = tf.reshape(clusters, (num_instances, block_rows, block_cols, channels))
        reshaped_covs = tf.reshape(clusters_covs, (num_instances, block_rows, block_cols, channels))
        transposed_clusters = tf.transpose(reshaped_clusters, [0, 3, 1, 2])
        transposed_covs = tf.sqrt(tf.transpose(reshaped_covs, [0, 3, 1, 2]))
        with tf.control_dependencies([training_op]):
            assign1 = tf.assign(templates_var, transposed_clusters)
            assign2 = tf.assign(weights_var, transposed_covs)
        return initializer, tf.group(assign1, assign2, name='gmm_init_assign')

