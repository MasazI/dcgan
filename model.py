#encoding: utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

# settings
import settings
FLAGS = settings.FLAGS

BATCH_SIZE = FLAGS.batch_size
Y_DIM = FLAGS.y_dim
Z_DIM = FLAGS.z_dim
GF_DIM = FLAGS.gf_dim
DF_DIM = FLAGS.df_dim
GFC_DIM = FLAGS.gfc_dim
DFC_DIM = FLAGS.dfc_dim
C_DIM = FLAGS.c_dim


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        # TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    print "=" * 100
    print shape
    print "=" * 100

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) + (1. - targets) * tf.log(1. - preds + eps)))


class BatchNorm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


class DCGAN:
    def __init__(self, dataset_name, checkpoint_dir):
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

    def inference(self, images, z):
        print "="*100
        print "images DCGAN inference:"
        print images.get_shape()
        print "="*100

        self.z_sum = tf.histogram_summary("z", z)

        # Generative
        print "generative"
        self.generator = Generative()
        self.G = self.generator.inference(z)

        # Discriminative
        print "discriminative from images"
        self.discriminator = Discriminative()
        self.D, self.D_logits = self.discriminator.inference(images)

        print "discriminative for sample from noize"
        self.sampler = self.generator.sampler(z)
        self.D_, self.D_logits_ = self.discriminator.inference(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        return images, self.D_logits, self.D_logits_, self.G_sum, self.z_sum, self.d_sum, self.d__sum

    def loss(self, logits, logits_):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        return self.d_loss_real, self.d_loss_fake, self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum, self.g_loss_sum, self.d_loss, self.g_loss

    def generate_images(self, z, row=8, col=8):
        images = tf.cast(tf.mul(tf.add(self.generator.sampler(z), 1.0), 127.5), tf.uint8)
        images = [image for image in tf.split(0, BATCH_SIZE, images)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
        image = tf.concat(1, rows)
        return tf.image.encode_png(tf.squeeze(image, [0]))


class Discriminative:
    def __init__(self):
        self.d_bn1 = BatchNorm(name='d_bn1')
        self.d_bn2 = BatchNorm(name='d_bn2')
        self.d_bn3 = BatchNorm(name='d_bn3')

    def inference(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        print "="*100
        print "image:"
        print image.get_shape()
        print "="*100

        h0 = lrelu(conv2d(image, DF_DIM, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, DF_DIM * 2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, DF_DIM * 4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, DF_DIM * 8, name='d_h3_conv')))

        print "="*100
        print "h3:"
        print h3.get_shape()
        print "="*100


        h4 = linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4


class Generative:
    def __init__(self):
        self.g_bn0 = BatchNorm(name='g_bn0')
        self.g_bn1 = BatchNorm(name='g_bn1')
        self.g_bn2 = BatchNorm(name='g_bn2')
        self.g_bn3 = BatchNorm(name='g_bn3')

    def inference(self, z):
        self.z_, self.h0_w, self.h0_b = linear(z, GF_DIM * 8 * 4 * 4, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, GF_DIM * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [BATCH_SIZE, 8, 8, GF_DIM * 4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(h1, [BATCH_SIZE, 16, 16, GF_DIM * 2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(h2, [BATCH_SIZE, 32, 32, GF_DIM * 1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(h3, [BATCH_SIZE, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    def sampler(self, z):
        tf.get_variable_scope().reuse_variables()

        # project `z` and reshape
        h0 = tf.reshape(linear(z, GF_DIM * 8 * 4 * 4, 'g_h0_lin'),
                        [-1, 4, 4, GF_DIM * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [BATCH_SIZE, 8, 8, GF_DIM * 4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [BATCH_SIZE, 16, 16, GF_DIM * 2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [BATCH_SIZE, 32, 32, GF_DIM * 1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [BATCH_SIZE, 64, 64, 3], name='g_h4')

        print "="*100
        print "h4:"
        print h4.get_shape()
        print "="*100

        return tf.nn.tanh(h4)
