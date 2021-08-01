from __future__ import print_function, division

import os
import click
from pix_lab.models.aru_net import ARUnet
from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.training.trainer import Trainer

@click.command()
@click.option('--path_list_train', default="......./lists/train.lst")
@click.option('--path_list_val', default="......./lists/val.lst")
@click.option('--output_folder', default="......./models/")
@click.option('--restore_path', default=None)
@click.option('--thread_num', default=1)
@click.option('--queue_capacity', default=4)
@click.option('--max_val', default=None)
@click.option('--scale_min', default=0.2)
@click.option('--scale_max', default=0.5)
@click.option('--scale_val', default=0.5)
@click.option('--seed', default=13)
@click.option('--steps_per_epoch', default=512)
@click.option('--epochs', default=100)
@click.option('--max_spat_dim', default=10000 * 10000)
@click.option('--gpu_device', default='0')
def run(
    path_list_train, 
    path_list_val,
    output_folder,
    restore_path,
    thread_num,
    queue_capacity,
    max_val,
    scale_min,
    scale_max,
    scale_val,
    seed,
    steps_per_epoch,
    epochs,
    max_spat_dim,
    gpu_device,
):
    # Since the input images are of arbitrarily size, the autotune will significantly slow down training!
    # (it is calculated for each image)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    # Images have to be gray scale images
    img_channels = 1
    # Number of output classes
    n_class = 3
    kwargs_dat = dict(
        batchsize_tr=1,
        batchsize_val=1,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_val=scale_val,
        max_val=int(max_val) if (max_val is not None) else None,
        affine_tr=True,
        one_hot_encoding=True,
        seed=seed,
    )
    data_provider = Data_provider_la(
        path_list_train,
        path_list_val,
        n_class,
        # threadNum=thread_num,
        queueCapacity=queue_capacity,
        kwargs_dat=kwargs_dat,
    )

    # choose between 'u', 'ru', 'aru', 'laru'
    model_kwargs = dict(model="ru")
    model = ARUnet(img_channels, n_class, model_kwargs=model_kwargs)
    opt_kwargs = dict(optimizer="rmsprop", learning_rate=0.001)
    cost_kwargs = dict(cost_name="cross_entropy")
    trainer = Trainer(model,opt_kwargs=opt_kwargs, cost_kwargs=cost_kwargs)
    trainer.train(
        data_provider,
        output_folder, 
        restore_path,
        batch_steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gpu_device=str(gpu_device),
        max_spat_dim=max_spat_dim
    )


if __name__ == '__main__':
    run()