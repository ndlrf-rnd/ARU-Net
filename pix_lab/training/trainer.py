from __future__ import print_function, division
from PIL import Image
import datetime
import os
import re
import sys
import time

import numpy
from clearml import Logger
from clearml import Task
import tqdm
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from .cost import get_cost
from .optimizer import get_optimizer
from tensorflow.python.client import device_lib

DEBUG = bool(os.environ.get('DEBUG', False))
RE_CHECKPOINT_PATH = "^(.+)[\\\/]([^\\\/\.0-9]+)([0-9]+)(\.[^.\\\/]+)?$"

def export_graph(sess, export_name, output_nodes=['output']):
    
    graph = tf.compat.v1.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        output_nodes  # The output node names are used to select the usefull nodes
    )
    # Finally we serialize and dump the output graph to the filesystem
    with tf.compat.v1.gfile.GFile(export_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

    print("Export Finished!")
    return export_name

def im2arr(im_in, normalize=False):
    im = numpy.array(im_in)
    if normalize:
        im = (im - numpy.min(im)) / ((numpy.max(im) - numpy.min(im)) or 1)
    if len(im.shape) == 2:
        return Image.fromarray(numpy.array(im * 255, dtype=numpy.uint8)).convert('L')
    elif len(im.shape) == 3:
        return Image.fromarray(numpy.array(im * 255, dtype=numpy.uint8)).convert('RGB')
    else:
        raise Exception('Cant convert to image: invalid input tensor dim')

class Trainer(object):
    """
    Trains a unet instance

    :param net: the arunet instance to train
    :param opt_kwargs: (optional) kwargs passed to the optimizer
    :param cost_kwargs: (optional) kwargs passed to the cost function

    """

    def __init__(self, net, opt_kwargs={}, cost_kwargs={}):
        self.net = net
        # Other params
        self.tgt = tf.compat.v1.placeholder("float", shape=[None, None, None, self.net.n_class])
        self.global_step = tf.compat.v1.placeholder("int64")
        self.opt_kwargs = opt_kwargs
        self.cost_kwargs = cost_kwargs
        self.cost_name=cost_kwargs.get("cost_name", "cross_entropy")
        self.histogram_freq=opt_kwargs.get('histogram_freq', 1)
        self.task = Task.init(project_name="document-segmentation/bd", task_name="aru-net train")
        # description='Baseline detection using *RU-Net'                   )
        self.logger = self.task.get_logger()

        self.cost = get_cost(self.net.logits, self.tgt, self.cost_kwargs)

    def _initialize(self, batch_steps_per_epoch=256, output_folder='./models/default/'):
        print('Initialize batch_steps_per_epoch', batch_steps_per_epoch)
        self.cost = get_cost(self.net.logits, self.tgt, self.cost_kwargs)
        self.optimizer, self.ema, self.learning_rate_node = get_optimizer(
            cost=self.cost,
            global_step=self.global_step,
            batch_steps_per_epoch=batch_steps_per_epoch,
            kwargs=self.opt_kwargs
        )
        
                        
        init = tf.compat.v1.global_variables_initializer()
        if not output_folder is None:
            output_folder = os.path.abspath(output_folder)
            if not os.path.exists(output_folder):
                print("Creating '{:}'".format(output_folder))
                os.makedirs(output_folder)
        return init
    
    def train(
        self, 
        data_provider,
        output_folder,
        restore_path=None, 
        batch_steps_per_epoch=256,
        epochs=100,
        gpu_device="0",
        max_pixels=4 * 1024 * 1024,
        image2tb_every_step=None,
        checkpoint_every_epoch=1,
        sample_every_steps=None,
    ):
        """
        Launches the training process
        :param data_provider:
        :param output_folder:
        :param restore_path:
        :param batch_size:
        :param batch_steps_per_epoch:
        :param epochs:
        :param keep_prob:
        :param gpu_device:
        :param max_pixels:
        :return:
        """
        
        print("Epochs: " + str(epochs))
        print("Batch Size Train: {}".format(data_provider.batchsize_tr))
        print("Batchsteps per Epoch {}".format(batch_steps_per_epoch))
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.abspath("logs/scalars/aru-net-{}".format(current_time))
        print("TF logs dir: {}".format(logdir))
        
        if output_folder is not None:
            output_folder = os.path.abspath(output_folder)
            if not os.path.exists(output_folder):
                print("Allocating '{}'".format(output_folder))
                os.makedirs(output_folder)
                
        if epochs == 0:
            return None
        
        init = self._initialize(
            batch_steps_per_epoch=batch_steps_per_epoch,
            output_folder=output_folder
        )
        
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        if gpu_device != 'cpu':
            session_conf.gpu_options.visible_device_list = gpu_device
        
        best_loss = None
        checkpoint_path = None
        with tf.compat.v1.Session(config=session_conf) as sess:
            # sess.run(train_summary_writer.init())
            sess.run(init)
            restore_path = restore_path or output_folder
            print('RESTORE PATH: {}'.format(restore_path))
            if restore_path is not None:
                latest_checkpoint_path = tf.train.latest_checkpoint(restore_path)
            
            start_from_epoch = 1
            if latest_checkpoint_path is not None:
                checkpoint_path = latest_checkpoint_path
                cp_state = tf.train.get_checkpoint_state(latest_checkpoint_path)
                print("Loading Checkpoint: '{}' ...".format(latest_checkpoint_path))
                self.net.restore(sess, latest_checkpoint_path)
                decomposed = re.findall(RE_CHECKPOINT_PATH, latest_checkpoint_path)
                if len(decomposed) > 0:
                    folder, prefix, epoch, suffix = decomposed[0]
                    start_from_epoch = int(epoch) + 1
                print('... checkpoint loaded, training will be started from epoch {}'.format(start_from_epoch))
            else:
                print('Clear model initialized, training will be started from epoch {}'.format(start_from_epoch))

            print("Starting busyloop...")
            
            predictor = None
            if sample_every_steps is not None:
                graph = tf.compat.v1.get_default_graph()
                predictor = graph.get_tensor_by_name('output:0')

            for epoch in range(start_from_epoch, epochs + 1):
                total_loss = 0
                lr = 0
                with tqdm.tqdm(range(0, batch_steps_per_epoch), unit="step", dynamic_ncols=True) as tepoch:
                    for step in tepoch:
                        global_step = int(((epoch - 1) * batch_steps_per_epoch) + step)
                        tepoch.set_description("Epoch {} [TRN: {}]".format(epoch, step))
                        dp_start = time.time()
                        batch_x, batch_tgt, paths = data_provider.next_data()
 
                        dp_time = time.time() - dp_start
                        if batch_x is None:
                            print("No Training Data available. Skip Training Path.")
                            break
                        train_start = time.time()
                        opt, loss, lr = sess.run(
                            [self.optimizer, self.cost, self.learning_rate_node],
                            feed_dict={
                                self.net.x: batch_x,
                                self.tgt: batch_tgt,
                                self.global_step: global_step,
                            },
                        )
                        
                        if self.cost_name is "cross_entropy_sum":
                            sh = batch_x.shape
                            loss = loss / (sh[1] * sh[2] * sh[0])
                        train_time = time.time() - train_start
                            
                        if (
                            loss > 2.0
                        ) or (
                            (sample_every_steps is not None) and ((global_step % sample_every_steps) == 0)
                        ):
        
                            prediction = sess.run(predictor, feed_dict={self.net.x: batch_x})
                            print(
                                batch_x.shape, numpy.min(batch_x), numpy.max(batch_x),
                                batch_tgt.shape, numpy.min(batch_tgt), numpy.max(batch_tgt),
                                prediction.shape, numpy.min(prediction), numpy.max(prediction),
                            )
                            self.logger.report_image(
                                "image",
                                ":input",
                                iteration=global_step,
                                image=im2arr(batch_x[0,:,:,0]),
                            )
                            
                            for channel in range(prediction.shape[3]):
                                self.logger.report_image(
                                    "image", 
                                    ":x:channel:{}".format(channel), 
                                    iteration=global_step, 
                                    image=im2arr(prediction[0,:,:,channel])
                                )
                            
                            for channel in range(batch_tgt.shape[3]):
                                self.logger.report_image(
                                    "image", 
                                    ":target:channel:{}".format(channel), 
                                    iteration=global_step, 
                                    image=im2arr(batch_tgt[0,:,:,channel])
                                )
                        
                        
                        self.logger.report_scalar(title=":loss:train", series='bd', iteration=global_step, value=loss)
                        
                        self.logger.flush()
                        
                        total_loss += loss
                        print('{} {} {}'.format(global_step, loss, ' '.join(paths)))
                        tepoch.set_postfix(
                            loss="{:.5f}".format(loss),
                            loss_agg="{:.5f}".format(total_loss / (step or 1)),
                            lr="{:.7f}".format(lr),
                            train_time="{:.3f}".format(train_time),
                            dp_time="{:.3f}".format(dp_time),
                            mpx_per_sec="{:.3f}".format(
                                ((batch_x.shape[1] * batch_x.shape[2]) / 1000000) / (train_time + dp_time)
                            )
                        )
                        

                # Save checkpoint
                if (output_folder is not None) and ((epoch % checkpoint_every_epoch) == 0):
                    checkpoint_base_path = os.path.join(output_folder, "model-")

                    checkpoint_path = '{}{:02d}'.format(checkpoint_base_path, epoch)
                    print('Saving checkpoint to: {}'.format(checkpoint_path))
                    self.net.save(sess, checkpoint_path)

                    graph_path = '{}{:02d}.pb'.format(checkpoint_base_path, epoch)
                    print('Saving TF graph to: {}'.format(graph_path))
                    export_graph(sess=sess, export_name=graph_path, output_nodes=['output'])


                print('Epoch {}, loss: {}, lr: {}'.format(epoch, total_loss, lr))
            if output_folder is not None:
                checkpoint_base_path = os.path.join(output_folder, "model-")
                graph_path = '{}{:02d}.pb'.format(checkpoint_base_path, int(epoch or start_from_epoch))
                if not os.path.exists(graph_path):
                    print('Saving TF graph to: {}'.format(graph_path))
                    export_graph(sess=sess, export_name=graph_path, output_nodes=['output'])
        data_provider.stop_all()
        print("Processing finished!")

            
        return checkpoint_path
