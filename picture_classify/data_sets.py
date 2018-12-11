import numpy as np
import os
import glob
from sklearn.utils import shuffle
import cv2


def load_train(train_path, img_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    print("读取训练图片...")
    for fields in classes:
        index = classes.index(fields)
        print("Now going to read {} files (Index:{})".format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls


class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def img_names(self):
        return self._img_names

    def cls(self):
        return self._cls

    def num_examples(self):
        return self._num_examples

    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            # assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()
    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])  # 1000,64,64,3;验证集200个
        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
        data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
        return data_sets
