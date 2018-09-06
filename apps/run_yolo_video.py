import os
current_path = os.path.dirname(os.path.abspath(__file__))
path_suffix = 'apps'
if current_path.endswith(path_suffix):
    parent_path = current_path.rsplit(path_suffix, 1)[0]
    os.chdir(parent_path)

from yolo_detect.yolo import YOLO
from yolo_detect.detect_video import detect_video

__author__ = 'sliu'

if __name__ == '__main__':
    video_file_name = '4云景ch0_CHANNEL0_20171011_15_45_40'
    video_path = 'input_data/videos/' + '%s.avi' % video_file_name
    out_path = 'output_data/' + '%s_output.avi' % video_file_name
    detect_video(YOLO(), video_path, out_path)