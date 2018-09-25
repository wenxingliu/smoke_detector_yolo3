import numpy as np
from track_vehicle.utils import paired_boxes_iou_too_small, bboxes_intersection_over_union, filter_small_bboxes,filter_imbalanced_bboxes

__author__ = 'sliu'


class BboxesManager:

    def __init__(self, interval, iou_threshold=0.8, bbox_size_threshold=0.01, ratio_w_and_h=0.2):
        self.interval = interval
        self.iou_threshold = iou_threshold
        self.bbox_size_threshold = bbox_size_threshold
        self.ratio_w_and_h = ratio_w_and_h
        self.frames_count = 0
        self.new_bboxes = None
        self.filtered_bboxes = np.array([])

    def clear_history(self):
        self.frames_count = 0
        self.new_bboxes = None
        self.filtered_bboxes = np.array([])

    def add_new_frame_exports(self, image_size, new_bboxes):
        if self.frames_count >= self.interval:
            self.clear_history()

        filtered_small_bboxes = self._filter_out_small_bboxes(new_bboxes, image_size)
        filtered_new_bboxes = self._filter_out_imbalanced_bboxes(image_size, filtered_small_bboxes)
        self.new_bboxes = self._dedupe_overlapping_bboxes_in_new_bboxes(filtered_new_bboxes)

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
        #new_boxes中多个bbox，两两计算overlap，不同object overlap太大的只取第一个ob，overlap都很小的话都计入filtered_new_bboxes
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
        #self.filtered_bboxes代表是已存在的之前一个frame里的bbox
        for i, existing_bbox in enumerate(self.filtered_bboxes):
            iou_list = list(map(lambda x: bboxes_intersection_over_union(x, existing_bbox), self.new_bboxes))
            max_iou = np.max(iou_list)
            #对上一个frame里每个box找出对应下一个frame里第mapped_index个box
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

    def _filter_out_small_bboxes(self, new_bboxes, image_size):
        if image_size:
            filtered_bboxes = filter_small_bboxes(image_size, new_bboxes, ratio_threshold=self.bbox_size_threshold)
            return filtered_bboxes
        else:
            return new_bboxes

    def _filter_out_imbalanced_bboxes(self, image_size, bboxes):
        if image_size:
            filtered_bboxes = filter_imbalanced_bboxes(bboxes, ratio_w_and_h=self.ratio_w_and_h)
            return filtered_bboxes
        else:
            return bboxes