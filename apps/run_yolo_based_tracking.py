import os

current_path = os.path.dirname(os.path.abspath(__file__))
path_suffix = 'apps'

if current_path.endswith(path_suffix):
    parent_path = current_path.rsplit(path_suffix, 1)[0]
    os.chdir(parent_path)

from shutil import rmtree
from yolo_detect.yolo import YOLO
from yolo_detect.detect_video import yolo_track_vehicles


__author__ = 'sliu'



def track_and_export_bboxes_in_all_videos_in_dir(videos_dir, root_out_dir, overwrite=False):

    video_files = [f for f in os.listdir(videos_dir)
                   if os.path.isfile(os.path.join(videos_dir, f)) and f.endswith('.avi')]

    yolo = YOLO()

    for video_file_name in video_files:
        video_file_path = os.path.join(videos_dir, video_file_name)
        video_num_str = video_file_name[:[s.isdigit() for s in video_file_name].index(False)]
        out_dir = root_out_dir + '//%s//' % video_num_str
        out_dir = os.path.join(root_out_dir, out_dir)

        if os.path.isdir(out_dir):
            if overwrite:
                rmtree(out_dir)
            else:
                continue

        yolo_track_vehicles(yolo, video_file_path, out_dir, 5)

        print('Process video %s' % video_file_name)

    yolo.close_session()


if __name__ == '__main__':
    video_path = 'C://dev//smoke_detector_yolo3//input_data//videos//60张坊进京ch0_2018_01_24_14_48_00.avi'
    output_dir = 'C://dev//smoke_detector_yolo3//output_data//60//'
    yolo = YOLO()
    yolo_track_vehicles(yolo, video_path, output_dir, 5)
    yolo.close_session()