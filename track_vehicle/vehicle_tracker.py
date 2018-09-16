from collections import deque
import numpy as np
from scipy.spatial.distance import cdist

from log_utils import save_numpy_file, save_image_to_file
from track_vehicle.utils import bboxes_pair_should_be_filtered_out, aug_bbox_range
from yolo_detect.utils import compute_bboxes_centerpoints

__author__ = 'sliu'


class VehicleTracker:

    def __init__(self, output_dir='', frame_index=0, min_frames_export=5):
        self.output_dir = output_dir
        self.frames_history = deque()
        self.tracked_objects = None
        self.popped_out_objects = None
        self.previous_frame_bboxes = None
        self.new_frame_bboxes = None
        self.frame_index = frame_index
        self.min_frames_export = min_frames_export
        self.number_of_exported_objects = 0
        self.average_moving_speed = 0

    def clear_history(self):
        self.frames_history = deque()
        self.tracked_objects = None
        self.popped_out_objects = None
        self.previous_frame_bboxes = None
        self.new_frame_bboxes = None

    def add_new_frame_to_tracker(self, new_frame, new_frame_bboxes, export=True):
        if not self.tracked_objects:
            self.initiate_with_first_image(new_frame, new_frame_bboxes)
        else:
            self.intake_new_frame_and_bboxes(new_frame, new_frame_bboxes)
            self.attribute_new_frame_bboxes_to_tracking_history()

        if self.popped_out_objects and export:
            self.export_tracking_objects()

    def initiate_with_first_image(self, new_frame, new_frame_bboxes):
        box_centerpoints = compute_bboxes_centerpoints(new_frame_bboxes)
        sorted_indices = np.argsort(box_centerpoints[:, 1])[::-1]
        self.new_frame_bboxes = new_frame_bboxes[sorted_indices]
        self.tracked_objects = deque([bbox] for bbox in new_frame_bboxes)
        self.frames_history.append(new_frame)

    def intake_new_frame_and_bboxes(self, new_frame, new_frame_bboxes):
        self.previous_frame_bboxes = np.array([object[-1] for object in self.tracked_objects])
        self.new_frame_bboxes = new_frame_bboxes

        self.frames_history.append(new_frame)
        if len(self.frames_history) > self.min_frames_export:
            self.frames_history.popleft()

    def attribute_new_frame_bboxes_to_tracking_history(self, distance_threshold=100, iou_threshold=0.5):
        num_boxes_0 = len(self.previous_frame_bboxes)
        num_boxes_1 = len(self.new_frame_bboxes)
        num_boxes_tracking = min(num_boxes_0, num_boxes_1)

        box_centerpoints_0 = compute_bboxes_centerpoints(self.previous_frame_bboxes)
        box_centerpoints_1 = compute_bboxes_centerpoints(self.new_frame_bboxes)

        dist_mx = cdist(box_centerpoints_0, box_centerpoints_1)
        closest_bbox_indices = np.argmin(dist_mx, axis=1)

        tracked_box_indices = set([])
        used_new_frame_bbox_indices = set([])

        for i, center_0 in enumerate(box_centerpoints_0):

            if i >= num_boxes_tracking:
                break

            mapped_index = closest_bbox_indices[i]
            used_new_frame_bbox_indices.add(mapped_index)
            center_1 = box_centerpoints_1[mapped_index]

            bbox_0 = self.previous_frame_bboxes[i]
            bbox_1 = self.new_frame_bboxes[mapped_index]

            if not bboxes_pair_should_be_filtered_out(bbox_0=bbox_0, bbox_1=bbox_1,
                                                      center_0=center_0, center_1=center_1,
                                                      distance_threshold=distance_threshold,
                                                      iou_threshold=iou_threshold):
                self.tracked_objects[i].append(bbox_1)
                tracked_box_indices.add(i)

        left_over_bboxes_in_new_frame = [bbox for i, bbox in enumerate(self.new_frame_bboxes)
                                         if i not in used_new_frame_bbox_indices]

        self._discard_lost_and_long_tracking_object(tracked_box_indices)
        self._append_new_tracking_object(left_over_bboxes_in_new_frame)

    def export_tracking_objects(self):
        for object_history in self.popped_out_objects:
            if len(object_history) >= self.min_frames_export:
                self._crop_and_export_bbox(object_history)

    def _discard_lost_and_long_tracking_object(self, tracked_box_indices):
        popped_out_objects = deque()
        tracked_objects = deque()

        for i, object_history in enumerate(self.tracked_objects):
            if (
                    i in tracked_box_indices
                    and self._tracked_object_not_have_enough_history(object_history)
            ):
                tracked_objects.append(object_history)
            elif self._is_bbox_leaving_camera(np.array(object_history)):
                popped_out_objects.append(object_history)

        self.popped_out_objects = popped_out_objects
        self.tracked_objects = tracked_objects

    def _crop_and_export_bbox(self, tracking_history, w_aug_factor=0.2, h_aug_factor=0.2):
        for i, bbox in enumerate(tracking_history[::-1]):
            corresponding_frame = self.frames_history[-(i+1)]
            image_size = corresponding_frame.size
            corresponding_frame = np.array(corresponding_frame, dtype='float32')
            crop_top, crop_left, crop_bottom, crop_right = aug_bbox_range(bbox,
                                                                          image_size,
                                                                          w_aug_factor,
                                                                          h_aug_factor)
            cropped_img = corresponding_frame[crop_top:crop_bottom, crop_left:crop_right, ]
            frame_num = self.frame_index - i
            image_file_name = '%d_frame_%d' % (self.number_of_exported_objects, frame_num)
            save_image_to_file(self.output_dir, image_file_name, cropped_img)

        self.number_of_exported_objects += 1

    def _append_new_tracking_object(self, new_objects):
        for new_object in new_objects:
            self.tracked_objects.append([new_object])

    def _tracked_object_not_have_enough_history(self, tracked_bboxes):
        return len(tracked_bboxes) < self.min_frames_export

    def _is_bbox_leaving_camera(self, tracked_bboxes):
        '''
        :param tracked_bboxes: array
        :return: boolean
        '''
        if len(tracked_bboxes) < self.min_frames_export:
            return False
        centerpoints = compute_bboxes_centerpoints(tracked_bboxes)
        moving_direction = np.mean([centerpoints[i, 1] >= centerpoints[(i+1), 1] for i in np.arange(len(centerpoints) - 1)])
        leaving = (centerpoints[0, 1] - centerpoints[-1, 1]) > 0 and moving_direction > 0.3
        return leaving

    def _update_vehicle_average_moving_speed(self):
        if self.frame_index % self.min_frames_export == 0:
            try:
                object_speeds = []
                for object_history in self.tracked_objects:
                    if len(object_history) >= self.min_frames_export:
                        centerpoints = compute_bboxes_centerpoints(np.array(object_history))
                        start_point = centerpoints[0][1]
                        end_point = centerpoints[-1][1]
                        object_speed = (start_point - end_point) / len(object_history)
                        object_speeds.append(object_speed)
                if object_speeds:
                    current_average = np.mean(object_speeds)
                    if self.average_moving_speed == 0:
                        self.average_moving_speed = current_average
                    else:
                        self.average_moving_speed = self.average_moving_speed * 0.5 + current_average * 0.5
            except:
                pass
