from __future__ import print_function, division

import datetime
import os
import re

import tqdm
import tensorflow as tf

from cost import get_cost
from optimizer import get_optimizer
from tensorflow.python.client import device_lib
from tensorflow.python.framework import graph_util
# from callbacks import CudaProfileCallback, LMSStatsLogger, LMSStatsAverage


# def get_callbacks(args):
#     callbacks = []

#     if hvd:
#         callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
#         callbacks.append(hvd.callbacks.MetricAverageCallback())

#     if args.nvprof:
#         callbacks.append(CudaProfileCallback(args.nvprof_epoch,
#                                              args.nvprof_start,
#                                              args.nvprof_stop))

#     if args.lms_stats:
#         stats_filename = os.path.join(args.output_dir,
#                                       generate_stats_name(args.model, "lms_stats"))
#         callbacks.append(LMSStatsLogger(stats_filename))

#     if args.lms_stats_average:
#         stats_filename = os.path.join(args.output_dir,
#                                       generate_stats_name(args.model, "lms_stats_average"))
#         lms = LMSStatsAverage(stats_filename,
#                               args.image_size,
#                               batch_size=args.batch_size,
#                               start_batch=args.lms_stats_warmup_steps)
#         callbacks.append(lms)

#     return callbacks

def export_graph(sess, export_name, output_nodes=['output']):
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        input_graph_def,  # The graph_def is used to retrieve the nodes
        output_nodes  # The output node names are used to select the usefull nodes
    )
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(export_name, "wb") as f:
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
#         self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

#         self.test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
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
          max_spat_dim=10000 * 10000
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
        :param max_spat_dim:
        :return:
        """
        print("Epochs: " + str(epochs))
        print("Batch Size Train: " + str(data_provider.batchsize_tr))
        print("Batchsteps per Epoc " + str(batch_steps_per_epoch))
        if not output_folder is None:
            checkpoint_path = os.path.join(output_folder, "model")
            
            
        if epochs == 0:
            return checkpoint_path

        init = self._initialize(batch_steps_per_epoch, output_folder)

        val_size = data_provider.size_val

        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        if gpu_device != 'cpu':
            session_conf.gpu_options.visible_device_list = gpu_device
            # session_conf.gpu_options.experimental.lms_enabled = True
        
        with tf.compat.v1.Session(config=session_conf) as sess:
#             file_writer = tf.summary.FileWriter(self.train_log_dir, sess.graph)
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
            
            bestLoss = 1.0 * 1000.0 * 1000.0
            shown_samples = 0
            
            for epoch in range(start_from_epoch, epochs):
                total_loss = 0
                lr = 0
                with tqdm.tqdm(
                    range(0, batch_steps_per_epoch),
                    unit="step",
                    leave=True
                ) as tepoch:
                    for step in tepoch:
                        tepoch.set_description("Epoch " + str(epoch + 1) + ' [TRN:' + str(step) + ']')
                        batch_x, batch_tgt = data_provider.next_data('train')
                        skipped = 0
                        if batch_x is None:
                            print("No Training Data available. Skip Training Path.")
                            break
                        while batch_x.shape[1] * batch_x.shape[2] > max_spat_dim:
                            batch_x, batch_tgt = data_provider.next_data('train')
                            skipped = skipped + 1
                            print(
                                "WARNING!!! Spatial Dimension of Training Data to high:",
                                batch_x.shape[1], 
                                'X',
                                batch_x.shape[2],
                                '=', 
                                batch_x.shape[1] * batch_x.shape[2],
                                '>', 
                                max_spat_dim,
                                "(MAX)",
                            )
                            if skipped > 100:
                                print("Spatial Dimension of Training Data to high. Aborting.")
                                return checkpoint_path
                        # Run training
#                         global_step = ( epoch + 1 ) * batch_steps_per_epoch + step
                        print('Running session', batch_x.shape, batch_tgt.shape, epoch)
                        _, loss, lr = sess.run(
                            [self.optimizer, self.cost, self.learning_rate_node],
                            feed_dict={
                                self.net.x: batch_x,
                                self.tgt: batch_tgt,
                                self.global_step: epoch
                            },
                        )
                        # shown_samples = shown_samples + batch_x.shape[0]
                        if self.cost_type is "cross_entropy_sum":
                            sh = batch_x.shape
                            loss = loss / (sh[1] * sh[2] * sh[0])
                        total_loss += loss

                        tepoch.set_postfix(
                            loss="{:.5f}".format(loss),
                            loss_agg="{:.5f}".format(total_loss / step),
                            lr="{:.7f}".format(lr)
                        )
                    
                if total_loss < bestLoss:
                    bestLoss = total_loss
                
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
                
                if ((epoch + 1) % self.test_every_epoch) == 0:
                    with tqdm.tqdm(
                        range(val_size), 
                        unit="batch",
                        leave=True
                    ) as vepoch:
                        total_loss = 0
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
                                }
                            )
                            if self.cost_type is "cross_entropy_sum":
                                sh = batch_x.shape
                                loss /= sh[1] * sh[2] * sh[0]
                            total_loss += loss
                            vepoch.set_postfix(
                                loss="{:.5f}".format(loss),
                                loss_agg="{:.5f}".format(total_loss / (step or 1)),
                            )
                        if val_size != 0:
                            data_provider.restart_val_runner()
                


            data_provider.stop_all()
            print("Optimization Finished! Best Val Loss: " + str(bestLoss))
            return checkpoint_path
