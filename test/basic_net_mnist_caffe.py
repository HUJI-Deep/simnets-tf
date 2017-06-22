import sys
sys.path.append(r'/home/elhanani/study/huji-deep/Generative-ConvACs/deps/simnets/python')

import caffe
import numpy as np
import pickle

def rshp_mex(x):
    return x.reshape(x.shape[0], x.shape[2], x.shape[3], 1, 1)

def prepare_weights_file(net):
    res_g = dict()
    res_g['sim_templates'] = net.params['sim1'][0].data
    res_g['sim_weights'] = net.params['sim1'][1].data
    res_g['mex1_offsets'] = rshp_mex(net.params['lv1_mex'][1].data)
    res_g['mex2_offsets'] = rshp_mex(net.params['lv2_mex'][1].data)
    res_g['mex3_offsets'] = rshp_mex(net.params['lv3_mex'][1].data)
    res_g['mex4_offsets'] = rshp_mex(net.params['lv4_mex'][1].data)
    res_g['mex5_offsets'] = rshp_mex(net.params['lv5_mex'][1].data)
    return res_g

def prepare_unified_output(net):
    res_a = dict()
    res_g = dict()
    res_a['sim'] = net.blobs['sim1'].data
    res_a['mex1'] = net.blobs['lv1_mex'].data
    res_a['mex1_pool'] = net.blobs['lv1'].data
    res_a['mex2'] = net.blobs['lv2_mex'].data
    res_a['mex2_pool'] = net.blobs['lv2'].data
    res_a['mex3'] = net.blobs['lv3_mex'].data
    res_a['mex3_pool'] = net.blobs['lv3'].data
    res_a['mex4'] = net.blobs['lv4_mex'].data
    res_a['mex4_pool'] = net.blobs['lv4'].data
    res_a['mex5'] = net.blobs['lv5_mex'].data

    res_g['sim_templates'] = net.params['sim1'][0].diff
    res_g['sim_weights'] = net.params['sim1'][1].diff
    res_g['mex1_offsets'] = rshp_mex(net.params['lv1_mex'][1].diff)
    res_g['mex2_offsets'] = rshp_mex(net.params['lv2_mex'][1].diff)
    res_g['mex3_offsets'] = rshp_mex(net.params['lv3_mex'][1].diff)
    res_g['mex4_offsets'] = rshp_mex(net.params['lv4_mex'][1].diff)
    res_g['mex5_offsets'] = rshp_mex(net.params['lv5_mex'][1].diff)

    return res_a, res_g

def main():
    net = caffe.Net(
        '/home/elhanani/study/huji-deep/Generative-ConvACs/exp/mnist/ght_model/train/net.prototxt',
        '/home/elhanani/study/huji-deep/Generative-ConvACs/exp/mnist/ght_model/train/ght_model_train_1_iter_250.caffemodel',
        caffe.TRAIN)
    res_g = prepare_weights_file(net)
    pickle.dump(res_g, open('caffe_weights.pkl', 'wb'))
    input_file = 'input.pkl'
    inp, lbl = pickle.load(open(input_file, 'rb'))
    net.forward(data=inp, label=lbl)
    net.backward()

    res_a, res_g = prepare_unified_output(net)
    pickle.dump([res_a, res_g], open('res_caffe.pkl', 'wb'))

if __name__ == '__main__':
    main()