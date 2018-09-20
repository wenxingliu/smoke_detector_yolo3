import numpy as np
from PIL import Image
import os
import cv2
from glob import glob

def acquire_data_and_label_from_image(path):
    train_data = []
    file_names = glob(path + '*.jpg')
    for file_name in file_names:
        file_path = os.path.abspath(file_name)
        image = Image.open(file_path)
        image = image.resize((224, 224))
        image_arr = np.array(image)
        train_data.append(image_arr)

    train_label = np.zeros(shape=[len(file_names), 2], dtype='int8')
    if path.endswith('No_smoke/'):
        train_label[:, 0] = 1
    else:
        train_label[:, 1] = 1
    return np.array(train_data), train_label





