import tensorflow.compat.v1 as tf


def get_optimizer(cost, global_step, batch_steps_per_epoch, kwargs={}):
    """
    Training: weight initialization: Xavier, optimizer:
    RMSprop, learning rate: 0.001, learning rate decay
    per epoch: 0.985, weight decay on the L2 norm:
    0.0005, exponential moving average on the model
    weights: 0.9995, mini batch size: 1 (due to memory limitations of the GPU), early stopping: none
    (trained for a fixed number of epochs)
    """
    optimizer_name = kwargs.get("optimizer", "rmsprop")
    learning_rate = kwargs.get("learning_rate", 0.001)
    lr_decay_rate = kwargs.get("lr_decay_rate", 0.985)
    ema_decay = kwargs.get("ema_decay", 0.9995)
    decay_epochs = kwargs.get("decay_epochs", 1)
    print(
        'global_step',global_step,
        'decay_epochs', decay_epochs,
        'batch_steps_per_epoch', batch_steps_per_epoch,
        'learning_rate', learning_rate, 
        'kwargs', kwargs
    )
    decay_steps = decay_epochs * batch_steps_per_epoch
    with tf.variable_scope('optimizer'):

        if optimizer_name is "momentum":

            momentum = kwargs.get("momentum", 0.9)

            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=lr_decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node, momentum=momentum)
        elif optimizer_name is "rmsprop":
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=lr_decay_rate,
                                                                 staircase=True)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_node)
        else:
            # Use Adam
            if not learning_rate is None:
                learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                global_step=global_step,
                                                                decay_steps=decay_steps,
                                                                decay_rate=lr_decay_rate,
                                                                staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node)
            else:
                optimizer = tf.train.AdamOptimizer()
                learning_rate_node = tf.constant(0.0)
        optimizer = optimizer.minimize(cost)

    ema = None
    if ema_decay > 0:
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op = ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        with tf.control_dependencies([optimizer]):
            optimizer = tf.group(maintain_averages_op)

    print("Optimizer: " + optimizer_name)
    if learning_rate is None:
        print("Learning Rate: " )
    else:
        print("Learning Rate: " + str(learning_rate))
    if ema_decay > 0:
        print("Use EMA: True")
    else:
        print("Use EMA: False")
    return optimizer, ema, learning_rate_node