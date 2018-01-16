import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_optimizer(name, lr=0.001, momentum=0.9):
    with tf.variable_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            if name == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            elif name == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=momentum)
            elif name == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
            else:
                raise ValueError("Unknown optimizer: {}".format(name))

    return optimizer


def get_activation(name):
    if name == "relu":
        return tf.nn.relu
    elif name == "relu6":
        return tf.nn.relu6
    elif name == "swish":
        return lambda x: tf.multiply(x, tf.nn.sigmoid(x))


def get_variables_to_train(trainable_scopes, logger):
    """Returns a list of variables to train.abs

        Returns:
            A list of variables to train by the optimizer.abs
    """

    if trainable_scopes is None or trainable_scopes == "":
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(",")]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    for var in variables_to_train:
        logger.info("vars to train > {}".format(var.name))

    return variables_to_train

