import numpy as np
from PIL import Image
import os
import cv2
from glob import glob

def get_image_array(file_name):
    file_path = os.path.abspath(file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224))
    image_arr = np.array(image)
    return image_arr

def acquire_data_and_label_from_image(path):
    train_data = []
    file_names = glob(path + '*.jpg')
    for file_name in file_names:
        image_arr = get_image_array(file_name)
        train_data.append(image_arr)

    train_label = np.zeros(shape=[len(file_names), 2], dtype='int8')
    if path.endswith('No_smoke/'):
        train_label[:, 0] = 1
    else:
        train_label[:, 1] = 1
    return np.array(train_data), train_label

def acquire_three_data_and_label_from_image(path):
    train_data = []
    file_names = glob(path + '*.jpg')
    i = 0
    while i < (len(file_names)):
        file_name1 = file_names[i]
        file_name2 = file_names[i+1]
        file_name3 = file_names[i+2]
        i += 3
        image_arr1 = get_image_array(file_name1)
        image_arr2 = get_image_array(file_name2)
        image_arr3 = get_image_array(file_name3)
        train_data.append([image_arr1, image_arr2, image_arr3])

    train_label = np.zeros(shape=[int(len(file_names)/3), 2], dtype='int8')
    if path.endswith('No_smoke/'):
        train_label[:, 0] = 1
    else:
        train_label[:, 1] = 1
    return np.array(train_data), train_label




