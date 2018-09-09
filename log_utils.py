import cv2
import os
import numpy as np


__author__ = 'sliu'


def save_numpy_file(dir_path, file_name, npy_obj):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    file_name = file_name + '.npy'
    file_path = os.path.join(dir_path, file_name)

    if os.path.isfile(file_path):
        print('overwrite %s\%s' % (dir_path, file_name))
        os.remove(file_name)

    np.save(file_path, npy_obj)


def load_numpy_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name + '.npy')
    return np.load(file_path)


def save_image_to_file(dir_path, file_name, image_obj):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if 'numpy' not in str(type(image_obj)):
        image_arr = np.array(image_obj, dtype='float32')
    else:
        image_arr = image_obj

    file_path = os.path.join(dir_path, file_name + '.jpg')

    if os.path.isfile(file_path):
        print('overwrite %s\%s' % (dir_path, file_name))
        os.remove(file_name)

    cv2.imwrite(file_path, image_arr)


def load_image(dir_path, file_name):
    image_file_path = os.path.join(dir_path, file_name)
    image = cv2.imread(image_file_path)
    return image