import os

current_path = os.path.dirname(os.path.abspath(__file__))
path_suffix = 'apps'

if current_path.endswith(path_suffix):
    parent_path = current_path.rsplit(path_suffix, 1)[0]
    os.chdir(parent_path)

from yolo_detect.yolo import YOLO
from yolo_detect.detect_video import yolo_detect_object_and_export_interim_outputs
from track_vehicle.track_vehicle import bounding_box_tracking, crop_bbox_from_image


__author__ = 'sliu'



def track_and_export_bboxes_in_all_videos_in_dir(videos_dir, root_out_dir, overwrite=False):
    yolo = YOLO()

    video_files = [f for f in os.listdir(videos_dir)
                   if os.path.isfile(os.path.join(videos_dir, f)) and f.endswith('.avi')]

    for video_file_name in video_files:
        video_file_path = os.path.join(videos_dir, video_file_name)
        video_num_str = video_file_name[:[s.isdigit() for s in video_file_name].index(False)]
        out_dir = root_out_dir + '//%s//' % video_num_str
        out_dir = os.path.join(root_out_dir, out_dir)

        if os.path.isdir(out_dir):
            if overwrite:
                os.rmdir(out_dir)
            else:
                continue

        print('Process video %s' % video_file_name)
        track_vehicle_and_export_bboxes(yolo, video_file_path, out_dir,
                                        top_N=5, track_num_frames=25,
                                        interval=5, export_every_n=5,
                                        w_aug_factor=0.2, h_aug_factor=0.2,
                                        save_cropped_image=True)

    yolo.close_session()


def track_vehicle_and_export_bboxes(yolo, video_path, output_dir,
                                    top_N=5, track_num_frames=20,
                                    interval=10, export_every_n=10,
                                    w_aug_factor=0.1, h_aug_factor=0.2,
                                    save_cropped_image=True):
    yolo_outputs, images_dict = yolo_detect_object_and_export_interim_outputs(yolo, video_path, output_dir)

    tracked_bboxes_dict = bounding_box_tracking(yolo_outputs, output_dir,
                                                top_N=top_N, track_num_frames=track_num_frames)

    cropped_images = crop_bbox_from_image(yolo_outputs, images_dict, tracked_bboxes_dict, output_dir,
                                          interval=interval, export_every_n=export_every_n,
                                          w_aug_factor=w_aug_factor, h_aug_factor=h_aug_factor,
                                          save_cropped_image=save_cropped_image)

    del images_dict, yolo_outputs, tracked_bboxes_dict

    return cropped_images


if __name__ == '__main__':
    video_path = 'C://dev//smoke_detector_yolo3//input_data//videos//41琉璃河ch0_CHANNEL0_20180108_11_56_50.avi'
    output_dir = 'C://dev//smoke_detector_yolo3//output_data//1//41琉璃河ch0_CHANNEL0_20180108_11_56_50//'
    yolo = YOLO()
    track_vehicle_and_export_bboxes(yolo, video_path, output_dir, True)
    yolo.close_session()