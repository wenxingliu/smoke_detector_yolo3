import numpy as np
from PIL import Image


__author__ = 'sliu'


def detect_image(yolo, input_img_path, output_img_path, exec_loop=False):
    continue_loop = True
    while continue_loop:
        try:
            image = Image.open(input_img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            import cv2
            cv2.imwrite(output_img_path, np.array(r_image, dtype='float32'))

        if exec_loop:
            input_str = input("Continue? (yes/[no])")
            if input_str.lower() != 'yes':
                continue_loop = False
        else:
            continue_loop = False

    yolo.close_session()