import sys
sys.path.append(r'/home/elhanani/study/huji-deep/install')
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from simnets import similarity
from simnets.ops.mex import mex, _mex_dims_helper, _expand_dim_specification
import sys
import pickle
from pprint import pprint

np.random.seed(1)

batch_size = 128
num_classes = 10
sim_kernel = 2
sim_channels = 32
mex_channels = sim_channels
epochs = 30

# input image dimensions
img_rows, img_cols = 28, 28

mnist = input_data.read_data_sets(one_hot=True, train_dir='/home/elhanani/tmp')

def sum_pooling_layer(x, pool_size=(2, 2)):
    # Average pooling doesn't take padding area into consideration when using padding='same'
    # If we wish to add padding, we must do so externally with ZeroPadding2D so we would
    # be able to compute the sum pooling layer from the average pooling layer.
    with tf.name_scope('sum_pool'):
        px, py = pool_size
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.avg_pool(x, blocks=[1, px, py, 1], strides=[1, px, py, 1],
                           padding='SAME', name='avg_pool')
        x = tf.transpose(x, [0, 3, 1, 2])
        res = tf.multiply(np.float32(px * py), x, name='sum_pool')
        graph = tf.get_default_graph() # type: tf.Graph
        graph.add_to_collection(tf.GraphKeys.ACTIVATIONS, res)
        return res


def dirichlet_init(alpha=1.0):
    def _dirichlet_init(shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = np.float32
        num_regions, num_instances, block_c, block_h, block_w = shape
        k = block_c * block_h * block_w
        # when given s as a size argument dirichlet function return an array with shape s + [k]
        # then we reshape the output to be of the same shape as the variable
        init_np = np.random.dirichlet([alpha] * k, size=(num_regions, num_instances)).astype(np.float32)
        init_np = np.log(init_np)
        init_np = init_np.reshape(shape)
        return tf.constant(init_np)
    return _dirichlet_init

def similarity_layer(input, num_instances, similarity_function='L2', strides=[1,1], blocks=[3,3], padding='SAME',
               normalization_term=False, normalization_term_fudge=0.001, ignore_nan_input=False,
               out_of_bounds_value=0, templates_initializer=tf.random_normal_initializer(),
               weights_initializer=tf.constant_initializer(100), name='Similarity'):
    with tf.name_scope(name):
        if isinstance(padding, str):
            if padding.upper() not in ['SAME', 'VALID']:
                raise ValueError('Padding must be one of SAME, VALID (or a list of ints)')
            padding = [0, 0] if padding == 'VALID' else [e//2 for e in blocks]
        input_shape = input.get_shape().as_list()
        with tf.name_scope('Similarity'):
            with tf.variable_scope('Similarity'):
                weights = tf.get_variable('weights', shape=(num_instances, input_shape[1], blocks[0], blocks[1]),
                                          dtype=tf.float32, initializer=weights_initializer)
                templates = tf.get_variable('templates', shape=(num_instances, input_shape[1], blocks[0], blocks[1]),
                                            dtype=tf.float32, initializer=templates_initializer)
            tf.summary.histogram('similarity/weights', weights)
            tf.summary.histogram('similarity/templates', templates)
            sim = similarity(input, templates, weights, similarity_function=similarity_function,
                                  blocks=blocks,
                                  strides=strides, padding=padding,
                                  normalization_term=normalization_term,
                                  normalization_term_fudge=normalization_term_fudge,
                                  ignore_nan_input=ignore_nan_input, out_of_bounds_value=out_of_bounds_value)
            tf.summary.histogram('similarity/activation', sim)
        graph = tf.get_default_graph() # type: tf.Graph
        graph.add_to_collection(tf.GraphKeys.ACTIVATIONS, sim)
        return sim

def mex_layer(input, num_instances, padding=[0, 0, 0], strides=[1, 1, 1], blocks=[1, 1, 1],
              epsilon=1.0, use_unshared_regions=True, shared_offset_region=[-1],
              unshared_offset_region=[-1], softmax_mode=False, blocks_out_of_bounds_value=0.0,
              blocks_round_down=True, normalize_offsets=False, offsets_initializer=dirichlet_init(), name='Mex'):
    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        blocks = _expand_dim_specification(input_shape, blocks)
        if isinstance(padding, str):
            if padding.upper() not in ['SAME', 'VALID']:
                raise ValueError('Padding must be one of SAME, VALID (or a list of ints)')
            padding = [0, 0, 0] if padding == 'VALID' else [e//2 for e in blocks]

        # Create a trainable weight variable for this layer.
        nregions = _mex_dims_helper(input_shape[1:], num_instances, blocks=blocks, padding=padding,
                                    strides=strides, use_unshared_regions=use_unshared_regions,
                                    shared_offset_region=shared_offset_region,
                                    blocks_round_down=blocks_round_down,
                                    unshared_offset_region=unshared_offset_region)
        with tf.variable_scope(input.op.name + 'Mex'):
            offsets = tf.get_variable('offsets', shape=(nregions, num_instances) + tuple(blocks),
                                      dtype=np.float32, initializer=offsets_initializer)

        if normalize_offsets:
            with tf.name_scope('NormalizeOffsets'):
                flattened = tf.reshape(offsets, [nregions*num_instances, -1])
                norm_terms = tf.reduce_logsumexp(flattened, axis=1, keep_dims=True)
                flattened_normed = flattened - norm_terms # Broadcast
                offsets = tf.reshape(flattened_normed, tf.shape(offsets))
        tf.summary.histogram('mex_offsets', offsets)
        mex_res = mex(input, offsets, num_instances, blocks=blocks, padding=padding,
                       strides=strides, use_unshared_regions=use_unshared_regions,
                       shared_offset_region=shared_offset_region,
                       blocks_out_of_bounds_value=blocks_out_of_bounds_value,
                       blocks_round_down=blocks_round_down,
                       unshared_offset_region=unshared_offset_region,
                       softmax_mode=softmax_mode, epsilon=epsilon)
        tf.summary.histogram('mex', mex_res)
        graph = tf.get_default_graph() # type: tf.Graph
        graph.add_to_collection(tf.GraphKeys.ACTIVATIONS, mex_res)

    return mex_res

def create_net(input, labels):
    tf.set_random_seed(1)
    net = input
    sim_kernel = 2
    sim_channels = 32
    mex_channels = sim_channels
    net = similarity_layer(input, sim_channels,
                     blocks=[2, 2], strides=[2, 2], similarity_function='L2',
                     normalization_term=True, padding=[2, 2], out_of_bounds_value=np.nan, ignore_nan_input=True,
                     normalization_term_fudge=1e-4, name='sim')
    i = 0
    while net.get_shape().as_list()[-2:] != [1, 1]:
        i += 1
        mex_channels *= 2
        unshared = 2
        curr_shape = net.get_shape().as_list()
        net = mex_layer(net, mex_channels,
                   blocks=[int(curr_shape[-3]), 1, 1], strides=[int(curr_shape[-3]), 1, 1],
                   softmax_mode=True, normalize_offsets=True,
                   use_unshared_regions=True, unshared_offset_region=[unshared], padding=[0, 0, 0], name='mex' + str(i))
        net = sum_pooling_layer(net, pool_size=(2, 2))

    curr_shape = net.get_shape().as_list()
    net = mex_layer(net, num_classes,
               blocks=[mex_channels, curr_shape[-2], curr_shape[-1]],
                    strides=[mex_channels, curr_shape[-2], curr_shape[-1]],
               softmax_mode=True, normalize_offsets=True,
               use_unshared_regions=True, unshared_offset_region=[1], name='mex5')
    net = tf.reshape(net, [-1, tf.reduce_prod(tf.shape(net)[1:])], name='flatten')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net, name='loss'))
    tf.summary.scalar('loss', loss)
    predictions = tf.nn.softmax(net, name='predictions')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), tf.float32))

    return loss, accuracy, predictions

def prepare_unified_output(activations, gradients):
    res_a = dict()
    res_g = dict()
    res_a['sim'] = activations['Similarity/Similarity_1:0']
    res_a['mex1'] = activations['Mex/Mex:0']
    res_a['mex1_pool'] = activations['sum_pool/sum_pool:0']
    res_a['mex2'] = activations['Mex_1/Mex:0']
    res_a['mex2_pool'] = activations['sum_pool_1/sum_pool:0']
    res_a['mex3'] = activations['Mex_2/Mex:0']
    res_a['mex3_pool'] = activations['sum_pool_2/sum_pool:0']
    res_a['mex4'] = activations['Mex_3/Mex:0']
    res_a['mex4_pool'] = activations['sum_pool_3/sum_pool:0']
    res_a['mex5'] = activations['Mex_4/Mex:0']

    res_g['sim_templates'] = gradients['gradients/Similarity/Similarity_1_grad/SimilarityParametersGrad:0']
    res_g['sim_weights'] = gradients['gradients/Similarity/Similarity_1_grad/SimilarityParametersGrad:1']
    res_g['mex1_offsets'] = gradients['gradients/Mex/NormalizeOffsets/Reshape_grad/Reshape:0']
    res_g['mex2_offsets'] = gradients['gradients/Mex_1/NormalizeOffsets/Reshape_grad/Reshape:0']
    res_g['mex3_offsets'] = gradients['gradients/Mex_2/NormalizeOffsets/Reshape_grad/Reshape:0']
    res_g['mex4_offsets'] = gradients['gradients/Mex_3/NormalizeOffsets/Reshape_grad/Reshape:0']
    res_g['mex5_offsets'] = gradients['gradients/Mex_4/NormalizeOffsets/Reshape_grad/Reshape:0']

    return res_a, res_g

# def main():
#     #with tf.device('/cpu:0'):
#     if True:
#         input_file = 'input.pkl'
#         inp, lbl = pickle.load(open(input_file, 'rb'))
#
#         caffe_weights = pickle.load(open('caffe_weights.pkl', 'rb'))
#         graph = tf.get_default_graph() # type: tf.Graph
#         input = tf.placeholder(tf.float32, shape=[None, 1, 28, 28], name='input')
#         labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
#
#         predictions, loss = create_net(input, labels)
#
#         variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#         variables = {e.name: e for e in variables}
#         assign_ops = [tf.assign(variables['Similarity/templates:0'], caffe_weights['sim_templates']),
#                       tf.assign(variables['Similarity/weights:0'], caffe_weights['sim_weights']),
#                       tf.assign(variables['Similarity/Similarity_1Mex/offsets:0'], caffe_weights['mex1_offsets']),
#                       tf.assign(variables['sum_pool/sum_poolMex/offsets:0'], caffe_weights['mex2_offsets']),
#                       tf.assign(variables['sum_pool_1/sum_poolMex/offsets:0'], caffe_weights['mex3_offsets']),
#                       tf.assign(variables['sum_pool_2/sum_poolMex/offsets:0'], caffe_weights['mex4_offsets']),
#                       tf.assign(variables['sum_pool_3/sum_poolMex/offsets:0'], caffe_weights['mex5_offsets'])]
#         assign_ops = tf.group(*assign_ops)
#
#         gradients = tf.gradients(loss, graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
#         gradients = {e.name: e for e in gradients}
#         pprint(gradients)
#         activations = graph.get_collection(tf.GraphKeys.ACTIVATIONS)
#         activations = {e.name: e for e in activations}
#
#         with tf.Session(graph=graph) as sess:
#             #sess.run(tf.global_variables_initializer())
#             sess.run(assign_ops)
#             act_d, grads_d = sess.run([activations, gradients], feed_dict={input: inp, labels: lbl})
#         res_a, res_g = prepare_unified_output(act_d, grads_d)
#         pickle.dump([res_a, res_g], open('res_tf.pkl', 'wb'))

def main():
    graph = tf.get_default_graph() # type: tf.Graph
    input = tf.placeholder(tf.float32, shape=[None, 1, 28, 28], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    loss, acc, predictions = create_net(input, labels)

    train = tf.train.AdamOptimizer(learning_rate=5*1e-3).minimize(loss)
    summs = tf.summary.merge_all()
    acc_summary = tf.summary.scalar('accuracy', acc)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('log', graph=graph)
        for i in range(3000):
            inp, lbls = mnist.train.next_batch(100)
            inp = inp.reshape(-1, 1, 28, 28) - 0.5
            l, _ = sess.run([loss, train], feed_dict={input:inp.astype(np.float64), labels: lbls})
            if i % 5 == 0:
                print('Iter {} Loss: {}'.format(i, l))
            if i % 10 == 0:
                writer.add_summary(sess.run(summs, feed_dict={input: inp.astype(np.float32), labels: lbls}), i)
            if i % 100 == 0:
                inp_tst, lbls_tst = mnist.test.next_batch(100)
                inp_tst = inp_tst.reshape(-1, 1, 28, 28) - 0.5
                acc_val, acc_pb = sess.run([acc, acc_summary], feed_dict={input: inp_tst.astype(np.float32), labels: lbls_tst})
                writer.add_summary(acc_pb, i)
                print('Accuracy:', acc_val)

if __name__ == '__main__':
    main()
