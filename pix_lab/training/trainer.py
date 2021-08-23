from __future__ import print_function, division
from PIL import Image
import datetime
import os
import re
import sys

import numpy as np

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
        self.checkpoint_every_epoch = opt_kwargs.get('checkpoint_every_epoch', 1)
        self.tgt = tf.compat.v1.placeholder("float", shape=[None, None, None, self.net.n_class])
        self.global_step = tf.compat.v1.placeholder("int64")
        self.opt_kwargs = opt_kwargs
        self.cost_kwargs = cost_kwargs
        self.cost_name=cost_kwargs.get("cost_name", "cross_entropy")
        self.histogram_freq=opt_kwargs.get('histogram_freq', 1)
        
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
                print("Allocating '{:}'".format(output_folder))
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
        max_pixels=10000 * 10000,
        image2tb_every_step=10,
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
        
        train_summary_writer = tf.compat.v2.summary.create_file_writer(logdir)
        
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

        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run(train_summary_writer.init())
            sess.run(init)
            restore_path = restore_path or output_folder
            print('RESTORE PATH: {}'.format(restore_path))
            if restore_path is not None:
                latest_checkpoint_path = tf.train.latest_checkpoint(restore_path)
            
            start_from_epoch = 1
            if latest_checkpoint_path is not None:
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
            
            graph = tf.compat.v1.get_default_graph()

            x = graph.get_tensor_by_name('inImg:0')
            predictor = graph.get_tensor_by_name('output:0')
            
            for epoch in range(start_from_epoch, epochs + 1):
                total_loss = 0
                lr = 0
                with tqdm.tqdm(range(0, batch_steps_per_epoch), unit="step", dynamic_ncols=True) as tepoch:
                    for step in tepoch:
                        global_step = int(((epoch - 1) * batch_steps_per_epoch) + step)
                        tepoch.set_description("Epoch {} [TRN: {}]".format(epoch, step))
                        batch_x, batch_tgt = data_provider.next_data()
                        skipped = 0
                        if batch_x is None:
                            print("No Training Data available. Skip Training Path.")
                            break
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


                        total_loss += loss

                        
                        with train_summary_writer.as_default():
                            
                            sess.run([
                                tf.compat.v2.summary.scalar('loss', loss, step=global_step),
                                tf.compat.v2.summary.scalar('lr', lr, step=global_step),
                                tf.compat.v2.summary.scalar('input/width', batch_x.shape[2], step=global_step),
                                tf.compat.v2.summary.scalar('input/height', batch_x.shape[1], step=global_step),
                                tf.compat.v2.summary.scalar('input/pixels', batch_x.shape[1] * batch_x.shape[2], step=global_step),
                            ])
                            if ((global_step + 1) % image2tb_every_step) == 0:
                                pred = sess.run(predictor, feed_dict={x: batch_x})
                                maxed_gt = (
                                    (np.expand_dims(
                                        np.argmax(
                                            batch_tgt[:,:,:,:],
                                            axis=3
                                        ),
                                        axis=3
                                    ) + 1)  * 64
                                ).astype(np.uint8)

                                maxed_out = (
                                    (np.expand_dims(
                                        np.argmax(
                                            pred[:,:,:,:],
                                            axis=3
                                        ),
                                        axis=3
                                    ) + 1)  * 64
                                ).astype(np.uint8)
                                print('batch_x', batch_x.shape, 'prediction', pred.shape)
                                sess.run([
                                    tf.compat.v2.summary.image('image/input', (batch_x * 255).astype(np.uint8), step=global_step),
                                    tf.compat.v2.summary.image('image/gt', maxed_gt, step=global_step),
                                    tf.compat.v2.summary.image('image/output', maxed_out, step=global_step),
                                    # tf.compat.v2.summary.trace_export(name="graph_trace", step=global_step, profiler_outdir=logdir),
                                ])
                        tepoch.set_postfix(**dict(
                            loss="{:.5f}".format(loss),
                            loss_agg="{:.5f}".format(total_loss / (step or 1)),
                            lr="{:.7f}".format(lr)
                        ))

                if (best_loss is None) or (total_loss < best_loss):
                    best_loss = total_loss

                # Save checkpoint
                if output_folder is not None:
                    if (epoch % self.checkpoint_every_epoch) == 0:
                        checkpoint_base_path = os.path.join(output_folder, "model-")

                        checkpoint_path = '{}{:02d}'.format(checkpoint_base_path, epoch)
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        self.net.save(sess, checkpoint_path)
                        # write_meta_graph=False

                        graph_path = '{}{:02d}.pb'.format(checkpoint_base_path, epoch)
                        print('Saving TF graph to: {}'.format(graph_path))
                        export_graph(sess=sess, export_name=graph_path, output_nodes=['output'])


                print('Epoch {}, loss: {}, lr: {}'.format(epoch, total_loss, lr))
        data_provider.stop_all()
        print("Processing finished!")

        if best_loss is not None:
            print(" Best train Loss: {}".format(best_loss))
            
        return checkpoint_path
