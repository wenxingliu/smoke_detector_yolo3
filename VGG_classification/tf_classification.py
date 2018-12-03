import os
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import inspect
from data_preprocess import *
from sklearn.utils import shuffle
from input_data_stream import *

__author__ = 'WangZe'

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            # path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.pardir()
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build_3(self, img1, img2, img3):
        f1 = self.build(img1)
        f2 = self.build(img2)
        f3 = self.build(img3)
        return f1, f2, f3

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # self.fc6 = self.fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # self.fc8 = self.fc_layer(self.relu7, "fc8")
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")
        #
        # self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))
        return self.pool5

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


class CNN:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # self.loss = None

        with tf.variable_scope("weights"):
            self.weights = {
                'conv1': tf.get_variable('conv1', [3, 3, 512, 512],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                'fc1': tf.get_variable('fc1', [3 * 3 * 512 * 3, 1024],
                                       initializer=tf.contrib.layers.xavier_initializer()),
                'pred': tf.get_variable('pred', [1024, 2], initializer=tf.contrib.layers.xavier_initializer()),
            }
        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable('conv1', [512, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [1024, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'pred': tf.get_variable('pred', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
            }

    def inference(self, vgg19, img1, img2, img3):
        pool5_1 = vgg19.build(img1)
        pool5_2 = vgg19.build(img2)
        pool5_3 = vgg19.build(img3)

        conv1_1 = self.conv_layers(pool5_1, 'conv1')
        conv1_2 = self.conv_layers(pool5_2, 'conv1')
        conv1_3 = self.conv_layers(pool5_3, 'conv1')
        concat = tf.concat([conv1_1, conv1_2, conv1_3], axis=3)
        fc1 = self.fc_layers(concat, 'fc1', activation='relu')
        pred = self.fc_layers(fc1, 'pred', activation='sigmoid')

        return pred

    # @staticmethod
    def compute_loss(self, prediction, labels):
        # labels = tf.one_hot(labels, 2)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,
                                                                        labels=tf.cast(labels, dtype=tf.float32)))

        return loss

    def train_step(self, loss):
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return optimizer

    def fc_layers(self, bottom, layers_name, activation):
        flatten = tf.reshape(bottom, [-1, self.weights[layers_name].get_shape().as_list()[0]])
        drop_out = tf.nn.dropout(flatten, 0.5)
        fc_layers = tf.matmul(drop_out, self.weights[layers_name]) + self.biases[layers_name]
        if activation == 'relu':
            return tf.nn.relu(fc_layers)
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(fc_layers)
        else:
            return fc_layers

    def conv_layers(self, bottom, layers_name):
        conv = tf.nn.bias_add(tf.nn.conv2d(bottom, self.weights[layers_name], strides=[1, 2, 2, 1], padding='VALID'),
                              self.biases[layers_name])
        relu = tf.nn.relu(conv)
        return relu

    def input_placeholder(self):
        x1 = tf.placeholder(tf.float32, shape=[self.batch_size, 224, 224, 3])
        x2 = tf.placeholder(tf.float32, shape=[self.batch_size, 224, 224, 3])
        x3 = tf.placeholder(tf.float32, shape=[self.batch_size, 224, 224, 3])
        y = tf.placeholder(tf.int64, shape=[self.batch_size, 2])
        return x1, x2, x3, y


def train(num_epochs):
    tf.reset_default_graph()
    # initialize the model
    vgg19 = Vgg19('./vgg19.npy')
    cnn = CNN(learning_rate=0.00001, batch_size=32)

    # get the data stream
    img1, img2, img3, label = read_and_decode(train_data_path + "train.tfrecords")
    img1_batch, img2_batch, img3_batch, label_batch = get_batch(img1, img2, img3, label, batch_size=cnn.batch_size)
    #inference
    x1, x2, x3, y = cnn.input_placeholder()
    pred = cnn.inference(vgg19, x1, x2, x3)
    loss = cnn.compute_loss(pred, y)
    train_step = cnn.train_step(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int64), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # get the test data stream
    test_img1, test_img2, test_img3, test_label = read_and_decode(test_data_path + "train.tfrecords")
    test_img1_batch, test_img2_batch, test_img3_batch, test_label_batch = get_batch(test_img1, test_img2,
                                                                                    test_img3, test_label,
                                                                                    batch_size=cnn.batch_size)
    # test data inference
    t1, t2, t3, ty = cnn.input_placeholder()
    test_pred = cnn.inference(vgg19, t1, t2, t3)
    test_loss = cnn.compute_loss(test_pred, ty)
    test_correct_prediction = tf.equal(tf.cast(tf.argmax(test_pred, 1), tf.int64), tf.argmax(ty, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    loss_step = []
    acc = []
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)
        print('training begin!')
        for i in range(num_epochs):
            # on train datasets
            image1, image2, image3, labels = sess.run([img1_batch, img2_batch, img3_batch, label_batch])
            feed = {x1: image1, x2: image2, x3: image3, y: labels}
            loss_, _, accuracy_ = sess.run([loss, train_step, accuracy], feed_dict=feed)
            loss_step.append(loss_)
            acc.append(accuracy_)
            if i % 5 == 0:
                print('epochs: %d' % i)
                print('**************loss: %.2f' % loss_)
                print('**************train acc: %.2f' % accuracy_)

            if i % 20 == 0:
                # on test datasets
                test_image1, test_image2, test_image3, test_labels = sess.run([test_img1_batch, test_img2_batch,
                                                                               test_img3_batch, test_label_batch])
                test_feed = {t1: test_image1, t2: test_image2, t3: test_image3, ty: test_labels}
                test_loss_, test_accuracy_ = sess.run([test_loss, test_accuracy], feed_dict=test_feed)
                print('**************test  acc: %.2f' % test_accuracy_)

            if i > 50:
                saver.save(sess, 'model/model', global_step=i)

        coord.request_stop()
        coord.join(threads)


# make the tfrecords file
train_data_path = 'C:\\Users\\昊天维业PC\\Desktop\\tf_cls\\data\\train_data\\'
test_data_path = 'C:\\Users\\昊天维业PC\\Desktop\\tf_cls\\data\\test_data\\'
classes = ('smoke', 'No_smoke')
if not os.path.exists(train_data_path + "train.tfrecords"):
    make_tfrecords(train_data_path, classes)
if not os.path.exists(test_data_path + "train.tfrecords"):
    make_tfrecords(test_data_path, classes)

train(num_epochs=1000)

