#encoding: utf-8
import os
import tensorflow as tf
import numpy as np
import time
from model import DCGAN
from train_operation import D_train_op
from train_operation import G_train_op
import dataset as traindataset
from PIL import Image

# settings
import settings
FLAGS = settings.FLAGS

BATCH_SIZE = FLAGS.batch_size
SAMPLE_SIZE = FLAGS.sample_size
Z_DIM = FLAGS.z_dim
IMAGE_SHAPE = [64, 64, 3]

DATA_DIR = FLAGS.data_directory
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement

EPOCHS = FLAGS.epochs

CSVFILE = FLAGS.csvfile

def train():
    with tf.Graph().as_default():
        # data
        dataset = traindataset.DataSet(DATA_DIR, SAMPLE_SIZE)
        # tfrecords inputs
        images, labels_t = dataset.csv_inputs(CSVFILE)

        z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')

        dcgan = DCGAN("test", "./checkpoint")
        images_inf, logits, logits_, G_sum, z_sum, d_sum, d__sum = dcgan.inference(images, z)
        d_loss_fake, d_loss_real, d_loss_real_sum, d_loss_fake_sum, d_loss_sum, g_loss_sum, d_loss, g_loss = dcgan.loss(logits, logits_)

        # trainable variables
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # train operations
        d_optim = D_train_op(d_loss, d_vars)
        g_optim = G_train_op(g_loss, g_vars)

        # generate images
        generate_images = dcgan.generate_images(z, 4, 4)

        # summary
        g_sum = tf.merge_summary([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
        d_sum = tf.merge_summary([z_sum, d_sum, d_loss_real_sum, d_loss_sum])
        #summary_op = tf.merge_all_summaries()

        # init operation
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        writer = tf.train.SummaryWriter("./logs", sess.graph_def)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # run
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # sampling
        sample_z = np.random.uniform(-1, 1, size=(SAMPLE_SIZE, Z_DIM))

        # sample images
        #sample_images = dataset.get_sample()

        counter = 1
        start_time = time.time()

        for epoch in xrange(EPOCHS):
            for idx in xrange(0, dataset.batch_idxs):
                #batch_images = dataset.create_batch()
                batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)

                # discriminative
                images_inf_eval, _, summary_str = sess.run([images_inf, d_optim, d_sum], {z: batch_z})
                writer.add_summary(summary_str, counter)

                #for i, image_inf in enumerate(images_inf_eval):
                #    print np.uint8(image_inf)
                #    print image_inf.shape
                #    #image_inf_reshape = image_inf.reshape([64, 64, 3])
                #    img = Image.fromarray(np.asarray(image_inf), 'RGB')
                #    print img
                #    img.save('verbose/%d_%d.png' % (counter, i))

                # generative
                _, summary_str = sess.run([g_optim, g_sum], {z: batch_z})
                writer.add_summary(summary_str, counter)

                # twice optimization
                _, summary_str = sess.run([g_optim, g_sum], {z: batch_z})
                writer.add_summary(summary_str, counter)

                errD_fake = sess.run(d_loss_fake, {z: batch_z})
                errD_real = sess.run(d_loss_real, {z: batch_z})
                errG = sess.run(g_loss, {z: batch_z})

                print("epochs: %02d %04d/%04d time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, dataset.batch_idxs,time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 10) == 1:
                    print("generate samples.")
                    generated_image_eval = sess.run(generate_images, {z: batch_z})
                    filename = os.path.join(FLAGS.sample_dir, 'out_%05d.png' % counter)
                    with open(filename, 'wb') as f:
                        f.write(generated_image_eval)
                counter += 1
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    train()
