from __future__ import print_function, division

import os
import click
from pix_lab.models.aru_net import ARUnet
from pix_lab.data_provider.data_provider_la import Data_provider_la
from pix_lab.training.trainer import Trainer

MODEL_TYPE_HELP = '''
* u - (U-Net). Vanilla U-Net architecture.'
* ru - (RU-Net). An RU-Net is an U-Net with residual blocks. That means, each of the 2 layer CNN blocks in replaced by a residual block.'
* aru (ARU-Net). An RU-Net incorporating the described spatial attention mechanism is called ARU-Net.' 
* laru - The LARU-Net is an ARU-Net with a separable MDLSTM9 layer at the lowest resolution to incorporate full spatial context.
'''

@click.command()
@click.option('--path_list', default="../lists/train.lst")
@click.option('--output_folder', default="../models/")
@click.option('--restore_path', default=None)
@click.option(
    '--model_name', '--model',
    default='aru',
    type=click.Choice(['ru', 'aru', 'laru'], case_sensitive=False), help=MODEL_TYPE_HELP.strip()
)
@click.option('--optimizer_name','--optimizer', 
    default='rmsprop', 
    type=click.Choice(['momentum', 'rmsprop'], case_sensitive=False)
)
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
@click.option('--act_name', '--act', 
    default='softmax', 
    type=click.Choice(['softmax', 'sigmoid', 'identity'], case_sensitive=False)
)
@click.option('--thread_num', '-j', default=1)
@click.option('--queue_capacity', '-q', default=1)
@click.option('--scale_min', default=0.2)
@click.option('--scale_max', default=0.5)
@click.option('--seed', default=13)
@click.option('--steps_per_epoch', default=256)
@click.option('--epochs', '-e', default=100)
@click.option('--max_pixels', '-b', default=(4 * 1024 * 1024))
@click.option('--gpu_device', default='0')
@click.option('--learning_rate', '--lr', '-l', default=0.001)
@click.option('--n_classes', '--n_class', '--num_classes', '--classes', '-n', default=2, type=int)
@click.option('--channels', default=1, type=int)
@click.option('--image2tb_every_step', default=None, type=int)
@click.option('--checkpoint_every_epoch', default=10, type=int)
@click.option('--sample_every_steps', default=None, type=int)
@click.option('--lr_decay_rate', default=0.985, type=float)
@click.option('--ema_decay', default=0.9995, type=float)
def run(
    path_list, 
    output_folder,
    restore_path,
    model_name,
    optimizer_name,
    cost_name,
    act_name,
    thread_num,
    queue_capacity,
    scale_min,
    scale_max,
    seed,
    steps_per_epoch,
    epochs,
    max_pixels,
    gpu_device,
    learning_rate,
    n_classes,
    channels,
    image2tb_every_step,
    checkpoint_every_epoch,
    sample_every_steps,
    lr_decay_rate,
    ema_decay,
):
    # https://stackoverflow.com/a/40126349
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

    data_provider = Data_provider_la(
        path_list,
        n_classes=n_classes,
        thread_num=thread_num,
        queue_capacity=queue_capacity,
        kwargs_dat=dict(
            batchsize_tr=1,
            channels=channels,
            scale_min=scale_min,
            scale_max=scale_max,
            affine_tr=True,
            # one_hot_encoding=True,
            seed=seed,
            max_pixels=max_pixels,
        ),
    )

    
    model = ARUnet(
        channels,
        n_classes + 1,  # pne for generated background layer
        model_kwargs=dict(
            model=model_name.lower().strip(),
        )
    )
 
    trainer = Trainer(
        model,
        opt_kwargs=dict(
            optimizer=optimizer_name.lower().strip(),
            learning_rate=learning_rate,
            # lr_decay_rate=lr_decay_rate,
            # ema_decay=ema_decay,
        ),
        cost_kwargs=dict(
            cost_name=cost_name.lower().strip(),
            act_name=act_name.lower().strip(),
        ),
    )
    print('trainer.train steps_per_epoch', steps_per_epoch)
    trainer.train(
        data_provider=data_provider,
        output_folder=output_folder, 
        restore_path=restore_path,
        batch_steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gpu_device=str(gpu_device),
        max_pixels=max_pixels,
        image2tb_every_step=image2tb_every_step,
        checkpoint_every_epoch=checkpoint_every_epoch,
        sample_every_steps=sample_every_steps,
    )


if __name__ == '__main__':
    run()
