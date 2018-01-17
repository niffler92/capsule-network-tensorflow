from functools import reduce
from operator import mul

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import common.utils as utils
import common.tf_utils as tf_utils
from common.tf_utils import get_optimizer


"""Reference
    Caps layer codes implemented in -
    https://github.com/naturomics/CapsNet-Tensorflow/blob/master/capsNet.py
"""
class BaseModel:
    def __init__(self, args):
        self.log = tf.logging
        self.args = args

        assert 'height' in args
        assert 'width' in args
        assert 'depth' in args
        assert 'num_classes' in args
        assert 'optimizer' in args
        assert 'learning_rate' in args
        assert 'batch_size' in args

        self.global_step = tf.Variable(0, dtype=tf.int32)

    def _create_placeholders(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.args['height'], self.args['width'], self.args['depth']], name="x")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
        self.y_onehot = tf.one_hot(self.y, depth=10, axis=1, dtype=tf.float32)

    def _create_network(self):
        raise NotImplementedError

    def _create_loss(self):
        raise NotImplementedError

    def _create_optimizer(self):
        """
        Args:
            optimizer (str): One of ["adam", "nesterov", "rmsprop", "adadelta"]
        """
        optimizer = get_optimizer(self.args['optimizer'], self.args['learning_rate'])
        variables_to_train = tf_utils.get_variables_to_train(None, self.log)

        if variables_to_train:
            self.train_op = slim.learning.create_train_op(
                self.total_loss,
                optimizer,
                global_step=self.global_step,
                variables_to_train=variables_to_train
            )
        else:
            self.log.info("Empty variables_to_train")
            self.train_op = tf.no_op()

    def _create_summaries(self):
        raise NotImplementedError

    def _show_current_model(self):
        tf_utils.show_all_variables()

    def build_graph(self):
        """ Building graph for the model """
        self._create_placeholders()
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._show_current_model()


class CapsNet(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)
        assert 'm_plus' in args
        assert 'm_minus' in args
        assert 'mask_with_y' in args

    def _create_network(self):
        batch_size = self.args['batch_size']

        with tf.variable_scope("conv1"):
            net = slim.conv2d(self.x, 256, 9, 1, padding='VALID')

        with tf.variable_scope("primary_caps"):
            primary_caps = CapsNet.caps_layer(x=net,
                             kernel_size=9,
                             stride=2,
                             num_outputs=32,
                             vec_len=8,
                             batch_size=batch_size,
                             is_routing=False,
                             layer_type='conv'
                             )
            #assert net.get_shape() == [batch_size, 1152, 8, 1]

        with tf.variable_scope("digit_caps"):
            digit_caps = CapsNet.caps_layer(x=primary_caps,
                             kernel_size=None,
                             stride=None,
                             num_outputs=10,
                             vec_len=16,
                             batch_size=batch_size,
                             is_routing=True,
                             layer_type='fc'
                             )

        with tf.variable_scope("masking"):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True) + 1e-10)
            #assert self.v_length.get_shape() == [batch_size, 10, 1, 1]
            softmax_v = tf.nn.softmax(self.v_length, dim=1)

            self.argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
            #assert self.argmax_idx.get_shape() == [batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(batch_size,))

            if not self.args['mask_with_y']:
                masked_v = []
                for bs in range(batch_size):
                    v = digit_caps[bs][self.argmax_idx[bs], :]
                    masked_v.append(tf.reshape(v, shape=[1, 1, 16, 1]))

                masked_v = tf.concat(masked_v, axis=0)
            else:
                masked_v = tf.multiply(tf.squeeze(digit_caps), tf.reshape(self.y_onehot, [-1, 10, 1]))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps), axis=2, keep_dims=True))

        with tf.variable_scope('decoder'):
            v_j = tf.reshape(masked_v, shape=(batch_size, -1))
            fc1 = slim.fully_connected(v_j, 512)
            assert fc1.get_shape() == [batch_size, 512]
            fc2 = slim.fully_connected(fc1, 1024)
            fc2.get_shape() == [batch_size, 1024]
            self.decoded = slim.fully_connected(fc2, 784, activation_fn=tf.nn.sigmoid)

    def _create_loss(self):
        bs = self.args['batch_size']
        # Margin Loss
        max_l = tf.square(tf.maximum(0., self.args['m_plus'] - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.args['m_minus']))
        #assert max_l.get_shape() == [bs, 10, 1, 1]
        max_l = tf.reshape(max_l, shape=[bs, -1])
        max_r = tf.reshape(max_r, shape=[bs, -1])

        t_c = self.y_onehot
        # element-wise multiplication
        l_c = t_c * max_l + self.args['lambda'] * (1 - t_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))

        # Reconstruction loss
        origin = tf.reshape(self.x, shape=[bs, -1])
        squared_err = tf.square(self.decoded - origin)
        self.reconstruction_loss = tf.reduce_mean(squared_err)

        # total loss
        self.total_loss = self.margin_loss + self.args['reg_scale'] * self.reconstruction_loss

        self.y_pred = self.argmax_idx
        corr_pred = tf.equal(self.y, self.y_pred)
        self.accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

    def _create_summaries(self):
        self.summary_train = tf.summary.merge(
            [
             tf.summary.scalar('train/accuracy', self.accuracy),
             tf.summary.scalar('train/margin_loss', self.margin_loss),
             tf.summary.scalar('train/reconstruction_loss', self.reconstruction_loss),
             tf.summary.scalar('train/total_loss', self.total_loss),
             tf.summary.image('reconstructed_img',
                              tf.reshape(self.decoded, shape=[self.args['batch_size'], 28, 28, 1])),
            ]
        )
        self.summary_valid = tf.summary.merge(
            [
             tf.summary.scalar('valid/accuracy', self.accuracy),
             tf.summary.scalar('vaiid/margin_loss', self.margin_loss),
             tf.summary.scalar('valid/reconstruction_loss', self.reconstruction_loss),
             tf.summary.scalar('valid/total_loss', self.total_loss),
             tf.summary.image('reconstructed_img',
                              tf.reshape(self.decoded, shape=[self.args['batch_size'], 28, 28, 1])),
            ]
        )

    @staticmethod
    def caps_layer(x,
                   kernel_size,
                   stride,
                   num_outputs,
                   vec_len,
                   batch_size,
                   is_routing,
                   layer_type
                   ):
        assert layer_type in ['conv', 'fc']
        # PrimaryCaps for conv
        # DigitCaps for fc

        if layer_type == 'conv':
            if not is_routing:
                #assert x.get_shape()[1:] == [20, 20, 256], x.get_shape()
                capsules = slim.conv2d(x, num_outputs * vec_len, kernel_size, stride, padding="VALID")
                capsules = tf.reshape(capsules, [batch_size, -1, vec_len, 1])
                capsules = CapsNet.squash(capsules)
                #assert capsules.get_shape()[1:] == [batch_size, 1152, 8, 1], capsules.get_shape()
                # bs * 6(after stride) * 6(after stride) * 256 / (bs * 8(vec_len) * 1) = 1152
                return capsules

        if layer_type == 'fc':
            if is_routing:
                x = tf.reshape(x, shape=[batch_size, -1, 1, x.shape[-2].value, 1])
                #assert x.get_shape() == [batch_size, 1152, 1, 8, 1]

                with tf.variable_scope('routing'):
                # b_ij: [bs, num_caps_in_layer(=1152), num_caps_in_layer+1, 1, 1]
                    #b_ij = tf.constant(np.zeros([batch_size, x.shape[1].value, num_outputs, 1, 1], dtype=np.float32))
                    b_ij = tf.constant(np.zeros([batch_size, 1152, num_outputs, 1, 1], dtype=np.float32))
                    capsules = CapsNet.routing(x, b_ij, 3, batch_size)
                    capsules = tf.squeeze(capsules, axis=1)

            return capsules

    @staticmethod
    def routing(x, b_ij, iterations, batch_size):
        """
        Args:
            x [bs, n_caps_in_layer_l=1152, 1, len(u_i)=8, 1]
        """

        weight = tf.get_variable('weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # tile for matmul
        x = tf.tile(x, [1, 1, 10, 1, 1])
        weight = tf.tile(weight, [batch_size, 1, 1, 1, 1])
        #assert x.get_shape() == [batch_size, 1152, 10, 8, 1]
        #assert weight.get_shape() == [batch_size, 1152, 10, 8, 16]

        u_hat = tf.matmul(weight, x, transpose_a=True)
        #assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

        for i in range(1, iterations+1):
            with tf.variable_scope("iter_{}".format(i)):
                c_ij = tf.nn.softmax(b_ij, dim=2)
                #assert c_ij.get_shape() == [batch_size, 1152, 10, 1, 1]

                if i == iterations:
                    s_j = tf.multiply(c_ij, u_hat)
                    #assert s_j.get_shape() == [batch_size, 1152, 10, 16, 1]
                    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                    #assert s_j.get_shape() == [batch_size, 1, 10, 16, 1]

                    v_j = CapsNet.squash(s_j)
                elif i < iterations:  # no backpropagations here
                    s_j = tf.multiply(c_ij, u_hat_stopped)
                    #assert s_j.get_shape() == [batch_size, 1152, 10, 16, 1]
                    s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                    #assert s_j.get_shape() == [batch_size, 1, 10, 16, 1]
                    v_j = CapsNet.squash(s_j)

                    v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1])
                    #assert v_j_tiled.get_shape() == [batch_size, 1152, 10, 16, 1]
                    u_product_v = tf.matmul(u_hat_stopped, v_j_tiled, transpose_a=True)
                    # because of transpose, [16, 1].T x [16, 1] = [1, 1]
                    #assert u_product_v.get_shape() == [batch_size, 1152, 10, 1, 1]

                    b_ij += u_product_v

        return v_j

    @staticmethod
    def squash(vec):
        epsilon = 1e-10  # to avoid zero division
        square_l2_norm = tf.reduce_sum(tf.square(vec), -2, keep_dims=True)
        squash_factor = square_l2_norm / (1 + square_l2_norm) / tf.sqrt(square_l2_norm + epsilon)

        return squash_factor * vec
