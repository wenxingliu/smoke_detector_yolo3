import numpy as np
from PIL import Image
import os
from glob import glob
import tensorflow as tf


def get_image_array(file_name):
    file_path = os.path.abspath(file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224))
    # image_arr = np.array(image)
    return image


def make_tfrecords(data_path, classes):
    writer = tf.python_io.TFRecordWriter(data_path + "train.tfrecords")
    for index, name in enumerate(classes):
        class_path = data_path + name + "/"
        file_names = glob(class_path + '*.jpg')
        i = 0
        while i < (len(file_names)):
            # train_data = []
            file_name1 = file_names[i]
            file_name2 = file_names[i + 1]
            file_name3 = file_names[i + 2]
            i += 3
            image_arr1 = get_image_array(file_name1)
            image_arr2 = get_image_array(file_name2)
            image_arr3 = get_image_array(file_name3)

            image_raw1 = image_arr1.tobytes()
            image_raw2 = image_arr2.tobytes()
            image_raw3 = image_arr3.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img1_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw1])),
                'img2_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw2])),
                'img3_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw3]))
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
    print("make tfrecords successfully !")


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img1_raw': tf.FixedLenFeature([], tf.string),
                                           'img2_raw': tf.FixedLenFeature([], tf.string),
                                           'img3_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img1 = tf.decode_raw(features['img1_raw'], tf.uint8)
    img2 = tf.decode_raw(features['img2_raw'], tf.uint8)
    img3 = tf.decode_raw(features['img3_raw'], tf.uint8)
    img1 = tf.reshape(img1, [224, 224, 3])
    img2 = tf.reshape(img2, [224, 224, 3])
    img3 = tf.reshape(img3, [224, 224, 3])

    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 2)
    return img1, img2, img3,  label


def get_batch(img1, img2, img3, label, batch_size):
    img1, img2, img3, label = tf.train.shuffle_batch([img1, img2, img3, label], batch_size=batch_size,
                                                 capacity=1000, min_after_dequeue=200)

    return img1, img2, img3, tf.reshape(label, [batch_size, 2])

# data_path = 'C:\\Users\\昊天维业PC\\Desktop\\tf_cls\\data\\train_data\\'
# classes = ('smoke', 'No_smoke')
# make_tfrecords(data_path, classes)

