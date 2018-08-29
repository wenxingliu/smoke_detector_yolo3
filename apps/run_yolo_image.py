from yolo_detect.detect_image import detect_image
from yolo_detect.yolo import YOLO

if __name__ == '__main__':

    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    detect_image(YOLO())
