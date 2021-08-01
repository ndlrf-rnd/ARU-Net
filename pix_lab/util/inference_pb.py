from __future__ import print_function, division

import time
import os
import tensorflow.compat.v1 as tf
import numpy as np
from scipy import misc
from pix_lab.util.util import load_graph
from PIL import Image, ImageOps
class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=1.0, mode='L', output_dir='output/'):
        self.graph = load_graph(path_to_pb)
        self.img_list = img_list
        self.scale = scale
        self.mode = mode
        self.output_dir = os.path.abspath(output_dir)
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except Exception as e:
            print('Failed to create output dir:', self.output_dir, 'due error:', e)

    def inference(self, print_result=True, gpu_device="0"):
        val_size = len(self.img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            
            # FIXME: move debug below to TF logs
            # print('Tensors:', [tensor.name for tensor in tf.get_default_graph().as_graph_def().node])
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            print("Start Inference")
            timeSum = 0.0
            for step in range(0, val_size):
                aTime = time.time()
                aImgPath = self.img_list[step]
                print(
                    "Image: {:} ".format(aImgPath))
                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                print(
                    "Resolution: h {:}, w {:} ".format(batch_x.shape[1],batch_x.shape[2]))
                # Run validation
                aPred = sess.run(predictor,
                                       feed_dict={x: batch_x})
                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime
                print(
                    "Update time: {:.2f} ms".format(curTime))
                if print_result:
                    n_class = aPred.shape[3]
                    channels = batch_x.shape[3]
                    for aI in range(0, n_class):
                        out_path = os.path.join(self.output_dir, str(step) + '-' + str(aI ) + '.png')
                        img = Image.fromarray((aPred[0,:, :,aI]  * 255).astype('uint8'), 'L')
                        img.save(out_path)

                    out_path = os.path.join(self.output_dir, str(step) + '.png')
                    img = Image.fromarray((batch_x[0, :, :, :].mean(axis=2)  * 255).astype('uint8'), 'L')
                    img = ImageOps.invert(img)
                    img.save(out_path)
            self.output_epoch_stats_val(timeSum/val_size)

            print("Inference Finished!")

            return None

    def output_epoch_stats_val(self, time_used):
        print(
            "Inference avg update time: {:.2f} ms".format(time_used))

    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        sImg = misc.imresize(aImg, scale, interp='bicubic')
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)

        return fImg
