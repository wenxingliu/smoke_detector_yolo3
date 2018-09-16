import numpy as np
from track_vehicle.utils import paired_boxes_iou_too_small, bboxes_intersection_over_union

__author__ = 'sliu'


class BboxesManager:

    def __init__(self, interval, iou_threshold=0.8):
        self.interval = interval
        self.iou_threshold = iou_threshold

        self.frames_count = 0
        self.new_bboxes = None
        self.filtered_bboxes = np.array([])

    def clear_history(self):
        self.frames_count = 0
        self.new_bboxes = None
        self.filtered_bboxes = np.array([])

    def add_new_frame_exports(self, new_bboxes):
        if self.frames_count >= self.interval:
            self.clear_history()

        self.new_bboxes = self._dedupe_overlapping_bboxes_in_new_bboxes(new_bboxes)

        if self.frames_count == 0:
            self.filtered_bboxes = np.array(self.new_bboxes)
        elif len(self.new_bboxes) > 0:
            self.filtered_bboxes = self._update_existing_objects_with_latest_bbox()

        self.frames_count += 1

    def ready_to_apply_tracking(self):
        return (self.frames_count >= self.interval) and (not self.no_objects_in_recent_history())

    def no_objects_in_recent_history(self):
        return len(self.filtered_bboxes) == 0

    def _dedupe_overlapping_bboxes_in_new_bboxes(self, new_bboxes):
        number_of_bboxes = len(new_bboxes)

        if number_of_bboxes <= 1:
            return new_bboxes

        kept_indices = set([])
        filtered_new_bboxes = []

        for i in np.arange(number_of_bboxes - 1):
            for j in np.arange(i + 1, number_of_bboxes):
                if paired_boxes_iou_too_small(new_bboxes[i], new_bboxes[j], self.iou_threshold):
                    if i not in kept_indices:
                        filtered_new_bboxes.append(new_bboxes[i])
                        kept_indices.add(i)
                    if j not in kept_indices:
                        filtered_new_bboxes.append(new_bboxes[j])
                        kept_indices.add(j)
                elif i not in kept_indices:
                    filtered_new_bboxes.append(new_bboxes[i])
                    kept_indices.add(i)

        return filtered_new_bboxes

    def _update_existing_objects_with_latest_bbox(self):
        paired_new_bboxes_indices = set([])
        filtered_bboxes = []
        for i, existing_bbox in enumerate(self.filtered_bboxes):
            iou_list = list(map(lambda x: bboxes_intersection_over_union(x, existing_bbox), self.new_bboxes))
            max_iou = np.max(iou_list)
            mapped_index = np.argmax(iou_list)

            # use bbox in new frame if overlaps with a bbox in previous frame
            if (max_iou >= self.iou_threshold) and (mapped_index not in paired_new_bboxes_indices):
                filtered_bboxes.append(self.new_bboxes[mapped_index])
                paired_new_bboxes_indices.add(mapped_index)
            # use bbox in previous frame if no overlapping bboxes in new frame
            else:
                filtered_bboxes.append(existing_bbox)

        new_objects = self._find_new_objects(paired_new_bboxes_indices)

        tracked_bboxes = np.array(filtered_bboxes + new_objects)
        return tracked_bboxes

    def _find_new_objects(self, paired_new_bbox_indices):
        new_objects = [bbox for i, bbox in enumerate(self.new_bboxes) if i not in paired_new_bbox_indices]
        return new_objects