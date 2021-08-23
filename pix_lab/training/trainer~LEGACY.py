from __future__ import print_function, division

import datetime
import os
import re
import sys

import numpy as np

import tqdm
import tensorflow as tf
# import tensorflow as tf2
# import tensorflow.compat.v1 as tf


from .cost import get_cost
from .optimizer import get_optimizer
from tensorflow.python.client import device_lib
# from tensorflow.python.framework import graph_util

DEBUG = bool(os.environ.get('DEBUG', False))

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
        
        # Logging
        # self.train_logdir = 'logs/gradient_tape/' + current_time + '/train'
        
        # Other params
        self.test_every_epoch = opt_kwargs.get('test_every_epoch', 10)
        self.checkpoint_every_epoch = opt_kwargs.get('checkpoint_every_epoch', 1)
        self.tgt = tf.compat.v1.placeholder("float", shape=[None, None, None, self.net.n_class])
        self.global_step = tf.compat.v1.placeholder(tf.int64)
        self.opt_kwargs = opt_kwargs
        self.cost_kwargs = cost_kwargs
        self.cost_type=cost_kwargs.get("cost_name", "cross_entropy")
        self.histogram_freq=opt_kwargs.get('histogram_freq', 1)
        
        
    def _initialize(self, batch_steps_per_epoch, output_folder):
        self.cost = get_cost(self.net.logits, self.tgt, self.cost_kwargs)
        self.optimizer, self.ema, self.learning_rate_node = get_optimizer(self.cost, self.global_step, batch_steps_per_epoch, self.opt_kwargs)

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
          max_pixels=10000 * 10000
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
        print("Batch Size Train: " + str(data_provider.batchsize_tr))
        print("Batchsteps per Epoc " + str(batch_steps_per_epoch))
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logdir = os.path.abspath("logs/scalars/aru-net-{}".format(current_time))
        print("TF LOGS DIR: {}".format(logdir))
        train_summary_writer = tf.summary.create_file_writer(logdir)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        
        if not output_folder is None:
            checkpoint_path = os.path.join(output_folder, "model")
            
            
        if epochs == 0:
            return checkpoint_path
        
        
        
        
        init = self._initialize(batch_steps_per_epoch, output_folder)

        val_size = data_provider.size_val

        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        if gpu_device != 'cpu':
            session_conf.gpu_options.visible_device_list = gpu_device
        
        with tf.compat.v1.Session(config=session_conf) as sess:
            sess.run(init)
            if restore_path is not None:
                latest_checkpoint_path = tf.train.latest_checkpoint(restore_path)
            else:
                restore_path = output_folder
                latest_checkpoint_path = tf.train.latest_checkpoint(output_folder)
            start_from_epoch = 0
            if latest_checkpoint_path is not None:
                cp_state = tf.train.get_checkpoint_state(latest_checkpoint_path)
                print('Loading Checkpoint: ' + latest_checkpoint_path + ' ...')
                self.net.restore(sess, latest_checkpoint_path)
                start_from_epoch = int(
                    re.findall(
                        "[0-9]+$",
                        os.path.splitext(latest_checkpoint_path)[0]
                    )[0] or 0
                ) 
                print(
                    'Done, will start from the next epoch after ' 
                    + str(start_from_epoch) 
                    + ': ' 
                    + str(start_from_epoch + 1) 
                    + ''
                )
            else:
                print('Clear model initialized')
                restore_path = None
            
            print("Starting busyloop")
            
            best_train_loss = None
            shown_samples = 0
            

            for epoch in range(start_from_epoch, epochs):
                total_train_loss = 0
                lr = 0
                with tqdm.tqdm(
                    range(0, batch_steps_per_epoch),
                    unit="step",
                    dynamic_ncols=True,
                ) as tepoch:
                    for step in tepoch:
                        tepoch.set_description("Epoch " + str(epoch + 1) + ' [TRN:' + str(step) + ']')
                        with train_summary_writer.as_default(step=(epoch * batch_steps_per_epoch) + step):
                            batch_x, batch_tgt = data_provider.next_data('train')
                            skipped = 0
                            if batch_x is None:
                                print("No Training Data available. Skip Training Path.")
                                break
                            # while batch_x.shape[1] * batch_x.shape[2] > max_pixels:
                            #     batch_x, batch_tgt = data_provider.next_data('train')
                            #     skipped = skipped + 1
                            #     print(
                            #         "WARNING!!! Spatial Dimension of Training Data to high:",
                            #         batch_x.shape[1], 
                            #         'X',
                            #         batch_x.shape[2],
                            #         '=', 
                            #         batch_x.shape[1] * batch_x.shape[2],
                            #         '>', 
                            #         max_pixels,
                            #         "(MAX)",
                            #     )
                            #     if skipped > 100:
                            #         print("Spatial Dimension of Training Data to high. Aborting.")
                            #         return checkpoint_path
                            # Run training
                            # global_step = ( epoch + 1 ) * batch_steps_per_epoch + step
                            #if DEBUG:
                            #    print('Running batch with train shape:', batch_x.shape, 'val shape:', batch_tgt.shape)

                            # print('batch_x', batch_x.shape, batch_x.dtype, batch_x.min(), batch_x.max())
                            # print('batch_tgt', batch_tgt.shape, batch_tgt.dtype)
                            # img = batch_x[0] # np.reshape(batch_x[0], (-1, batch_x.shape[1], batch_x.shape[2], 1))
                            # print('img', img.shape, img.dtype)
                            # floating point values in the range [0,1], or
                            # uint8 values in the range [0,255]


                            # current_stp = (batch_steps_per_epoch * epoch) + step
    #                         with train_summary_writer.as_default():
    #                         tf.summary.image("Target", batch_x_imgs, step=step)
    #                         train_summary_writer.image("Input", batch_x_imgs, step=step)
                            # tf.summary.image("Target", batch_tgt, step=current_stp)
#   writer = tf.summary.create_file_writer("/tmp/mylogs/session")

                              with train_summary_writer.as_default():
                                tf.summary.scalar("my_metric", 0.5, step=step)
                            all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
                            writer_flush = writer.flush()
                            _, loss, lr = sess.run(
                                [train_summary_writer.init(), self.optimizer, self.cost, self.learning_rate_node],
                                feed_dict={
                                    self.net.x: batch_x,
                                    self.tgt: batch_tgt,
                                    self.global_step: epoch
                                },
                            )
#                             tf.summary.scalar("loss", loss)
                            # shown_samples = shown_samples + batch_x.shape[0]
                            if self.cost_type is "cross_entropy_sum":
                                sh = batch_x.shape
                                loss = loss / (sh[1] * sh[2] * sh[0])
                            total_train_loss += loss
                            pf = dict(
                                loss="{:.5f}".format(loss),
                                loss_agg="{:.5f}".format(total_train_loss / (step or 1)),
                                lr="{:.7f}".format(lr)
                            )
                            tepoch.set_postfix(**pf)

                if (best_train_loss is None) or (total_train_loss < best_train_loss):
                    best_train_loss = total_train_loss


                # Save checkpoint
                if output_folder is not None:
                    if (epoch + 1) % self.checkpoint_every_epoch == 0:
                        checkpoint_pathAct = checkpoint_path + str(epoch + 1)
                        print(
                            'Saving checkpoint to: ' 
                            + checkpoint_pathAct 
                            + ' and TF graph to: ' 
                            + checkpoint_pathAct
                        )
                        self.net.save(sess, checkpoint_pathAct)
                        export_graph(
                            sess=sess,
                            export_name=checkpoint_pathAct + '.pb',
                            output_nodes=['output'],
                        )

                best_val_loss = None
                total_val_loss = 0

                # Evaluate
                if ((epoch + 1) % self.test_every_epoch) == 0:
                    with tqdm.tqdm(
                        range(val_size), 
                        unit="step",
                        dynamic_ncols=True,
                        # file=sys.stdout,
                        # leave=False,
                    ) as vepoch:
                        total_val_loss = 0
                        vepoch.set_description("Epoch " + str(epoch + 1) + ' [VAL]')
                        for step in vepoch:
                            batch_x, batch_tgt = data_provider.next_data('val')
                            if batch_x is None:
                                print("No Validation Data available. Skip Validation Path.")
                                break
                            # Run validation
                            loss, aPred = sess.run(
                                [
                                    self.cost,
                                    self.net.predictor
                                ], 
                                feed_dict={
                                    self.net.x: batch_x,
                                    self.tgt: batch_tgt
                                },
                            )
                            if self.cost_type is "cross_entropy_sum":
                                sh = batch_x.shape
                                loss = loss / (sh[1] * sh[2] * sh[0])
                            total_val_loss += loss
                            pf = dict(
                                loss="{:.5f}".format(loss),
                                loss_agg="{:.5f}".format(total_val_loss / (step or 1)),
                            )
                            vepoch.set_postfix(**pf)
                        if val_size != 0:
                            data_provider.restart_val_runner()
                if (best_val_loss is None) or (best_val_loss < best_val_loss):
                    best_val_loss = total_val_loss


            data_provider.stop_all()
            print("Processing finished!")
                  
            if best_train_loss is not None:
                print(" Best train Loss: {}".format(best_train_loss))
            if best_val_loss is not None:
                print("Best val loss: {}".format(best_val_loss))
            
            return checkpoint_path
