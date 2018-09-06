import os

current_path = os.path.dirname(os.path.abspath(__file__))
path_suffix = 'track_vehicle'

if current_path.endswith(path_suffix):
    parent_path = current_path.rsplit(path_suffix, 1)[0]
    os.chdir(parent_path)

import cv2
import numpy as np
from PIL import Image
from yolo_detect.yolo import YOLO
from track_vehicle.track_vehicle import compute_pairwise_bbox_match, track_all_bboxes_for_n_frames
from track_vehicle.utils import  plot_tracking_target, generate_colors_dict


__author__ = 'sliu'


def run_yolo_based_tracking(video_path, processed_img_output_path, boxed_img_output_path):
    yolo = YOLO()
    vid = cv2.VideoCapture(video_path)

    yolo_outputs = {}
    images_dict = {}

    count = 0
    return_value = True
    while return_value:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        processed_image, outputs_json = yolo.detect_image(image, True)
        if len(outputs_json) > 0:
            yolo_outputs[count] = outputs_json
            images_dict[count] = processed_image
            # output_file_name = img_output_file_path + 'processed_frame_%d.jpg' % count
            # cv2.imwrite(output_file_name, np.array(processed_image, dtype='float32'))

            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    yolo.close_session()

    paired_bboxes_list = compute_pairwise_bbox_match(yolo_outputs, top_N=10)
    tracked_bboxes_dict = track_all_bboxes_for_n_frames(paired_bboxes_list,
                                                        start_frame_index=0,
                                                        look_at_num_frames=len(yolo_outputs),
                                                        track_num_frames=20)
    colors = generate_colors_dict(50)

    previous_index = -50
    for i, track_path in tracked_bboxes_dict.items():
        if i - previous_index <= 10:
            continue
        previous_index = i
        plot_tracking_target(track_path, images_dict, colors, every_n_frames=5, output_file_path=boxed_img_output_path)
