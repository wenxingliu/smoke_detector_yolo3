import os
from PIL import Image

from yolo_detect.yolo_tensorflow import YOLO


__author__ = 'sliu'

current_path = os.path.dirname(os.path.abspath(__file__))
path_suffix = 'apps'
if current_path.endswith(path_suffix):
    parent_path = current_path.rsplit(path_suffix, 1)[0]
    os.chdir(parent_path)


if __name__ == '__main__':
    yolo = YOLO()

    input_img_path = 'input_data/45_frame_440.jpg'
    output_img_path = 'output_data/45_frame_440.jpg'

    image = Image.open(input_img_path)

    yolo.detect_image(image, output_img_path)
