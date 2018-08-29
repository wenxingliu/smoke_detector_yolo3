import numpy as np
from yolo import YOLO
from PIL import Image

def detect_img(yolo):
    while True:
        try:
            input_img_path = input('Input image filename:')
            image = Image.open(input_img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            import cv2
            output_img_path = input('Output image filename:')
            cv2.imwrite(output_img_path, np.array(r_image, dtype='float32'))
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    detect_img(YOLO())
