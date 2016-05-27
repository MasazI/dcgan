#encoding: utf-8
import os
import random
from glob import glob
import tensorflow as tf
import numpy as np

# settings
import settings
FLAGS = settings.FLAGS
BATCH_SIZE = FLAGS.batch_size
IS_CROP = FLAGS.is_crop
IS_RESIZE = FLAGS.is_resize
IMAGE_SIZE = FLAGS.image_size
TRAIN_SIZE = FLAGS.train_size

IMAGE_HEIGHT_ORG = FLAGS.image_height_org
IMAGE_WIDTH_ORG = FLAGS.image_width_org
IMAGE_DEPTH_ORG = FLAGS.image_depth_org

NUM_THREADS = FLAGS.num_threads

class DataSet:
    def __init__(self, datadir, sample_size):
        self.data = glob(os.path.join("./data", datadir, "*.jpg"))
        print("dataset number: %d" % (len(self.data)))
        self.data_num = len(self.data)
        self.sample_size = sample_size
        self.imagedata = ImageData()
        self.batch_idxs = min(len(self.data), TRAIN_SIZE) / BATCH_SIZE

    def get_sample(self):
        sample_files = self.data[0:self.sample_size]
        samples = [self.imagedata.getImage(sample_file, is_crop=IS_CROP, is_resize=IS_RESIZE) for sample_file in sample_files]
        self.sample_images = np.array(samples).astype(np.float32)
        return self.sample_images

    def create_batch(self):
        batche_files = random.sample(self.data, BATCH_SIZE)
        batches = [self.imagedata.getImage(batch_file, is_crop=IS_CROP, is_resize=IS_RESIZE) for batch_file in batche_files]
        return np.array(batches).astype(np.float32)

    def _generate_image_and_label_batch(self, image, label, min_queue_examples):
        '''
        imageとlabelのmini batchを生成
        '''
        num_preprocess_threads = NUM_THREADS
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue=min_queue_examples
        )
    
        # Display the training images in the visualizer
        #tf.image_summary('images', images, max_images=BATCH_SIZE)
        return images, tf.reshape(label_batch, [BATCH_SIZE])

    def csv_inputs(self, csv):
        print csv
        filename_queue = tf.train.string_input_producer([csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, label = tf.decode_csv(serialized_example, [["path"], [0]])

        label = tf.cast(label, tf.int64)
        jpeg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpeg, channels=3)
        image = tf.cast(image, tf.float32)
        image.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, IMAGE_DEPTH_ORG])

        image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)
        
        min_fraction_of_examples_in_queue = 0.4
        #min_fraction_of_examples_in_queue = 1
        min_queue_examples = int(100 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)
    
        return self._generate_image_and_label_batch(image, label, min_queue_examples)

    def tfrecords_inputs(self, tfrecords_file):
        '''
        create inputs
        '''
        print tfrecords_file
        filename_queue = tf.train.string_input_producer([tfrecords_file]) # ここで指定したepoch数はtrainableになるので注意
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
    
        # parse single example
        features = tf.parse_single_example(
            serialized_example,
            features = {
                "image_raw": tf.FixedLenFeature([], dtype=tf.string),
                "height": tf.FixedLenFeature([], dtype=tf.int64),
                "width": tf.FixedLenFeature([], dtype=tf.int64),
                "depth": tf.FixedLenFeature([], dtype=tf.int64),
                "label": tf.FixedLenFeature([], dtype=tf.int64),
            }
        )   
 
        # image
        _image = tf.decode_raw(features['image_raw'], tf.uint8)
        _image = tf.reshape(_image, [IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, IMAGE_DEPTH_ORG])
        #_image.set_shape([IMAGE_HEIGHT_ORG, IMAGE_WIDTH_ORG, IMAGE_DEPTH_ORG])   
 
        image = tf.cast(_image, tf.float32) * (1. /255) - 0.5
    
        # dense label
        label = tf.cast(features['label'], tf.int32)

        #resized_image = tf.image.central_crop(reshaped_image, )
        #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
        resized_image = tf.image.resize_images(_image, IMAGE_SIZE, IMAGE_SIZE)
        #float_image = tf.image.per_image_whitening(resized_image)
    
        min_fraction_of_examples_in_queue = 0.4
        #min_fraction_of_examples_in_queue = 1
        min_queue_examples = int(100 * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)
    
        return self._generate_image_and_label_batch(resized_image, label, min_queue_examples)


class ImageData:
    def getImage(self, filename, is_crop=False, is_resize=False, resize_height=IMAGE_SIZE, resize_width=IMAGE_SIZE):
        import cv2
        image = cv2.imread(filename)
        # bgr to rgb
        if is_resize:
            image = cv2.resize(image, (resize_height, resize_width))

        # transform bgr to rgb
        b, g, r = cv2.split(image)  # get b,g,r
        image = cv2.merge([r, g, b])  # switch it to rgb

        if is_crop:
            reshaped_image = tf.cast(image, tf.float32)
            height = IMAGE_SIZE
            width = IMAGE_SIZE
            # crop
            image = tf.random_crop(reshaped_image, [height, width])

        return image
