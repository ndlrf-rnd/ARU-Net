import io
import os
import sys
import time
import re
import click
from PIL import Image
# from PIL import ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.training.cost import get_cost
from pix_lab.util.util import load_graph
DEBUG=False

def debug(*vals):
    if DEBUG:
        print(*vals)
@click.command()
@click.option('--input', '--path_list_val', default="../lists/val.lst")
@click.option('--output', default=None)
@click.option('--path_to_pb', default="./demo_nets/model100_ema.pb")
@click.option('--cost_name', '--cost', 
    default='cross_entropy', 
    type=click.Choice([
        'cross_entropy',
        'cross_entropy_sum',
        'dice',
        'dice_mean',
        'mse',
        'mse_mean',
        'nse',
        'nse_mean',
        'combined',
    ], case_sensitive=False)
)
@click.option(
    '--act_name', '--act', 
    default='softmax', 
    type=click.Choice(['softmax', 'sigmoid', 'identity'], case_sensitive=False)
)
@click.option('--thread_num', '-j', default=1)
@click.option('--queue_capacity', '-q', default=1)
@click.option('--max_val', '-v', default=None, type=int)
@click.option('--scale_min', default=0.5)
@click.option('--scale_max', default=0.5)
@click.option('--n_class', '--num_classes', '--classes', '-n', default=3, type=int)
@click.option('--channels', default=1, type=int)
@click.option('--seed', default=13)
@click.option('--max_pixels', '-b', default=(4 * 1024 * 1024))
@click.option('--gpu_device', default='0')
@click.option('--batch_size', default=1)
@click.option('--plt-out/--no-plt-out', default=False)
@click.option('--report_name', default='report.tsv')
def validate(
    input,
    output,
    path_to_pb,
    cost_name,
    act_name,
    thread_num,
    queue_capacity,
    max_val,
    scale_min,
    scale_max,
    n_class,
    channels,
    seed,
    max_pixels,
    gpu_device,
    batch_size,
    plt_out,
    report_name,
):
    # https://stackoverflow.com/a/40126349
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    output_dir = os.path.abspath(output if output else '.')
    if not os.path.exists(output_dir):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except Exception as e:
            print('Failed to create output dir:', output_dir, 'due error:', e)
            exit(-1)

    output_tsv_path = os.path.join(output_dir, report_name)
    with io.open(output_tsv_path, 'w') as outf:
        outf.write(
            '{}\n'.format(
                '\t'.join([
                    'global_step',
                    'batch',
                    'loss',
                    'filename',
                    'model',
                    'pixels',
                    'width',
                    'height',
                    'channels',
                    'time_sec'
                ])
            ))
    data_provider = Data_provider_la(
        input,
        input,
        n_classes=n_class,
        queue_capacity=queue_capacity,
        thread_num=thread_num,
        kwargs_dat=dict(
            val_mode=True,
            batchsize_tr=batch_size,
            batchsize_val=batch_size,
            channels=channels,
            scale_min=scale_min,
            scale_max=scale_max,
            max_val=None,
            affine_tr=False,
            one_hot_encoding=True,
            seed=seed,
            max_pixels=max_pixels,
        ),
    )
        
    val_size = max_val or data_provider.size_val
    debug('VAL data records: {}\n'.format(val_size))
    graph = load_graph(path_to_pb)
    session_conf = tf.compat.v1.ConfigProto()
    session_conf.gpu_options.visible_device_list = gpu_device
    with tf.compat.v1.Session(graph=graph, config=session_conf) as sess:
        x = graph.get_tensor_by_name('inImg:0')
        log = graph.get_tensor_by_name('logits:0')
        predictor = graph.get_tensor_by_name('output:0')
        tgt = tf.compat.v1.placeholder("float", shape=[None, None, None, n_class])
        cost = get_cost(
            log,
            tgt,
            dict(
                cost_name=cost_name.lower().strip(),
                act_name=act_name.lower().strip(),
            )
        )
        debug("Start validation\n")

        total_loss = 0
        global_step = 0
        time_val_step = time.time()
        for step in range(0, val_size):
            tstart = time.time()
            batch_x, batch_tgt, batch_paths = data_provider.next_data()
            if batch_x is None:
                debug("No Validation Data available. Skip Validation Path.\n")
                break
            # Run validation
            loss, aPred = sess.run(
                [cost, predictor],
                feed_dict={
                    x: batch_x,
                    tgt: batch_tgt
                }
            )
            total_loss += loss
            

            n_class = aPred.shape[3]
            channels = batch_x.shape[3]

            for im_idx, bp in zip(range(aPred.shape[0]), batch_paths):
                out_str = '{}\n'.format(
                    '\t'.join([
                        '{}'.format(v) for v in [
                            global_step,
                            step,
                            loss,
                            bp,
                            path_to_pb,
                            batch_x.shape[1] * batch_x.shape[2],
                            batch_x.shape[1],
                            batch_x.shape[2],
                            batch_x.shape[3],
                            (time.time() - tstart) / aPred.shape[0]
                        ]
                    ])
                )
                with io.open(output_tsv_path, 'a') as outf:
                    outf.write(out_str)
                sys.stdout.write(out_str)
                global_step += 1
                if output:
                    fn = os.path.splitext(os.path.basename(bp))[0]
                    sp = os.path.join(output_dir,fn)
                    for class_idx in range(0, n_class):
                        img = Image.fromarray((aPred[im_idx,:, :,class_idx]  * 255).astype('uint8'), 'L')
                        img.save('{}_{}.jpeg'.format(sp, class_idx))

                    img = Image.fromarray((batch_x[im_idx, :, :, :].mean(axis=2)  * 255).astype('uint8'), 'L')
                    # img = ImageOps.invert(img)
                    img.save('{}.jpeg'.format(sp))
        
        total_loss = total_loss / val_size
        time_used = time.time() - time_val_step
        debug("VAL: Average loss: {:.8f}, time used: {:.2f}\n".format(total_loss, time_used))
        # sys.stdout.write('{}\n'.format(total_loss))
        data_provider.stop_all()
        debug("Validation Finished!")
        return total_loss

if __name__ == '__main__':
    validate()