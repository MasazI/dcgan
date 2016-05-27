#encoding: utf-8
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The size of sample images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("data_directory", "face", "Directory of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for crop image, False for not nothing [False]")
flags.DEFINE_boolean("is_resize", True, "True for resize image, False for nothing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("tfrecords", '/home/deep/source/tfrecords_tool/face/output/face.tfrecords', "tfrecords path for training.")
flags.DEFINE_string("csvfile", '/home/deep/workspace/dcgan/list_ero.txt', "cvs file path for training.")
flags.DEFINE_integer("image_height_org", 178, "original image height")
flags.DEFINE_integer("image_width_org", 218, "original image width")
flags.DEFINE_integer("image_depth_org", 3, "original image depth")
flags.DEFINE_integer("num_threads", 4, "number of threads using queue")

flags.DEFINE_integer("y_dim", None, "dimension of dim for y")
flags.DEFINE_integer("z_dim", 100, "dimension of dim for Z for sampling")
flags.DEFINE_integer("gf_dim", 64, "dimension of generative filters in first conv layer")
flags.DEFINE_integer("df_dim", 64, "dimension of discriminative filters in first conv layer")
flags.DEFINE_integer("gfc_dim", 1024, "dimension of generative units for full-connect-layer")
flags.DEFINE_integer("dfc_dim", 1024, "dimension of discriminative units for full-connect-layer")
flags.DEFINE_integer("c_dim", 3, "dimension of image channel")

flags.DEFINE_boolean('log_device_placement', False, 'where to log device placement.')



