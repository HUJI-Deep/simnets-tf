import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    k,s,r,p = 3,255,1,'SAME'

    images = np.arange(256*256).reshape(1,256,256,1).astype(np.int32)
    image = np.squeeze(images)
    sess = tf.Session()
    with tf.Session() as sess:
        image_tf = tf.constant(images)
        patches = tf.extract_image_patches(images=images, ksizes=[1, k, k, 1], strides=[1, s, s, 1], rates=[1, r, r, 1], padding=p).eval()
        patches = patches.reshape(patches.shape[1]*patches.shape[2],-1)
        for i in range(patches.shape[0]):
            img = np.zeros_like(image)
            img.reshape(-1)[patches[i].ravel()] = 255
            img = img.astype(np.uint8)
            plt.clf()
            plt.imshow(img)
            plt.show()
            #plt.pause(0.05)
if __name__ == '__main__':
    main()