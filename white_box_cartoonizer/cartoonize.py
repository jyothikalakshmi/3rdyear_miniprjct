"""
Internal code snippets were obtained from https://github.com/SystemErrorWang/White-box-Cartoonization/

For it to work tensorflow version 2.x changes were obtained from https://github.com/steubk/White-box-Cartoonization 
"""
import os
import sys
import cv2
import numpy as np


import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from white_box_cartoonizer import network

from white_box_cartoonizer import guided_filter

class WB_Cartoonize:
    def __init__(self, weights_dir, gpu):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.load_model(weights_dir, gpu)
        print("Weights successfully loaded")
    
    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                            interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        return image

    def load_model(self, weights_dir, gpu):
        try:
            tf.disable_eager_execution()
        except:
            pass

        tf.compat.v1.reset_default_graph()

        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        if gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            device_count = {'GPU': 1}
        else:
            gpu_options = None
            device_count = {'GPU': 0}

        config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)

        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(weights_dir))

    def infer(self, image):
        image = self.resize_crop(image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)

        ## Session Run
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})

        ## Post Process
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output

if __name__ == '__main__':
    gpu = len(sys.argv) < 2 or sys.argv[1] != '--cpu'
    wbc = WB_Cartoonize(os.path.abspath('white_box_cartoonizer/saved_models'), gpu)
    img = cv2.imread('white_box_cartoonizer/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cartoon_image = wbc.infer(img)
    import matplotlib.pyplot as plt
    plt.imshow(cartoon_image)
    plt.show()
