import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
def main():
    d = sys.argv[1]
    if not os.path.exists(d):
        os.mkdir(d)

    k,s,r,p = 11,4,1,'SAME'
    dim = 16
    images = np.arange(dim*dim).reshape(1,dim,dim,1).astype(np.int32)
    image = np.squeeze(images)
    sess = tf.Session()
    fig = plt.figure()
    plt.gray()
    with tf.Session() as sess:
        image_tf = tf.constant(images)
        patches = tf.extract_image_patches(images=images, ksizes=[1, k, k, 1], strides=[1, s, s, 1], rates=[1, r, r, 1], padding=p).eval()
        patches = patches.reshape(patches.shape[1]*patches.shape[2],-1)
        for i in range(patches.shape[0]):
            img = np.zeros_like(image)
            img.reshape(-1)[patches[i].ravel()] = 255

            img = img.astype(np.uint8)
            plt.clf()
            plt.imshow(img, interpolation='none')
            plt.title('d = {dim}, k = {k}, s = {s}, p = {p}'.format(**locals()))
            plt.xticks(np.arange(0,dim)-0.5)
            plt.yticks(np.arange(0,dim)-0.5)
            plt.gca().grid(which='major', axis='both', linestyle='-', color='w')
            fig.show()
            plt.savefig('{}/frame{:0>8}.png'.format(d, i))
if __name__ == '__main__':
    main()
