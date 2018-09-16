from collections import defaultdict
import cv2
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image

from track_vehicle.vehicle_tracker import VehicleTracker
from track_vehicle.bboxes_manager import BboxesManager
from log_utils import save_numpy_file, save_image_to_file

__author__ = 'sliu'


def detect_video(yolo, video_path, output_path=""):
    if not os.path.isfile(video_path):
        raise IOError("Video path not valid")
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
            save_image_to_file(output_path, '%d_%s' % (curr_fps, str(accum_time)), image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def yolo_track_vehicles(yolo, video_path, output_dir, min_frames_export=3, interval=3):

    object_tracker = VehicleTracker(output_dir=output_dir, min_frames_export=min_frames_export, interval=interval)
    bboxes_manager = BboxesManager(interval=interval)

    vid = cv2.VideoCapture(video_path)
    curr_fps = 0
    return_value, frame = vid.read()
    while return_value:
        image = Image.fromarray(frame)
        new_frame, bboxes_info = yolo.detect_image(image, True)
        new_bboxes = bboxes_info.get('bboxes', [])

        bboxes_manager.add_new_frame_exports(new_bboxes=new_bboxes)

        if bboxes_manager.ready_to_apply_tracking():
            object_tracker.frame_index = curr_fps
            object_tracker.add_new_frame_to_tracker(new_frame=new_frame,
                                                    new_frame_bboxes=bboxes_manager.filtered_bboxes)
        elif bboxes_manager.no_objects_in_recent_history():
            object_tracker.clear_history()
            bboxes_manager.clear_history()

        curr_fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        del return_value, frame, new_frame, bboxes_info
        return_value, frame = vid.read()

    yolo.close_session()


def yolo_detect_object_and_export_interim_outputs(yolo, video_path, output_dir, min_frames_export=5):

    object_tracker = VehicleTracker(output_dir=output_dir, min_frames_export=min_frames_export)

    vid = cv2.VideoCapture(video_path)

    yolo_outputs = defaultdict()
    images_dict = defaultdict()

    curr_fps = 0
    return_value, frame = vid.read()
    while return_value:
        image = Image.fromarray(frame)
        processed_image, bboxes_info = yolo.detect_image(image, True)

        if len(bboxes_info) > 0:
            yolo_outputs[curr_fps] = bboxes_info
            images_dict[curr_fps] = processed_image
            save_image_to_file(output_dir, 'processed_%d' % curr_fps, processed_image)
            object_tracker.frame_index = curr_fps
            object_tracker.add_new_frame_to_tracker(new_frame=processed_image,
                                                    new_frame_bboxes=bboxes_info['bboxes'])
        else:
            object_tracker.clear_history()

        curr_fps += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        del return_value, frame, processed_image, bboxes_info
        return_value, frame = vid.read()

    # save to file
    save_numpy_file(output_dir, 'yolo_detection_outputs', np.array([yolo_outputs]))

    return yolo_outputs, images_dict