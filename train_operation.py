#encoding: utf-8
import tensorflow as tf
import numpy as np

# settings
import settings
FLAGS = settings.FLAGS

BETA1 = FLAGS.beta1
LR = FLAGS.learning_rate


def D_train_op(d_loss, d_vars):
    d_optim = tf.train.AdamOptimizer(LR, beta1=BETA1).minimize(d_loss, var_list=d_vars)
    return d_optim


def G_train_op(g_loss, g_vars):
    g_optim = tf.train.AdamOptimizer(LR, beta1=BETA1).minimize(g_loss, var_list=g_vars)
    return g_optim