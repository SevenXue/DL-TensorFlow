from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile

# 原图像尺度为32*32,信息部分位于图像的中央，以中心剪裁后的图片尺寸
IMAGE_SIZE = 24

#类别，训练集，测试集大小
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_quence):

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    #定义一个reader，每次能从文件中读取固定字节数
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_quence)

    #解码操作，读二进制文件，把字符串中的字节转换为数值向量
    record_bytes = tf.decode_raw(value, tf.uint8)

    #从一堆tensor对象中截取一个slice
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])

    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )

    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image, [height, width])
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch%d.bin' % i)
                     for i in xrange(1, 6)]

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('failed to find file' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    float_image = tf.image.per_image_standardization(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size)
