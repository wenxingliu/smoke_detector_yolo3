import numpy as np

__author__ = 'sliu'



def log_detection_outputs_to_json(bboxes, centerpoints, scores, labels, image_size, vehicle_box_indices):
    if (len(bboxes) > 0) and (len(vehicle_box_indices) > 0):
        json_file = {"bboxes": bboxes[vehicle_box_indices],
                     'centerpoints': centerpoints[vehicle_box_indices],
                     "scores": scores[vehicle_box_indices],
                     "labels": labels[vehicle_box_indices],
                     "image_size": image_size}
        return json_file
    else:
        return {}


def find_bbox_corners(box, image_size):
    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image_size[0], np.floor(right + 0.5).astype('int32'))
    return top, left, bottom, right


# vectorized
def compute_bboxes_centerpoints(bboxes):
    # left, right; top, bottom
    center_points = np.array([(bboxes[:, 1] + bboxes[:, 3]) / 2, (bboxes[:, 0] + bboxes[:, 2]) / 2]).T
    return center_points


def non_negative_coord_suppress_bboxes(bboxes, image_size):
    num_boxes = len(bboxes)
    top, left, bottom, right = np.rollaxis(bboxes, -1)
    top = np.maximum(np.zeros(num_boxes), np.floor(top + 0.5).astype('int32'))
    left = np.maximum(np.zeros(num_boxes), np.floor(left + 0.5).astype('int32'))

    bottom = np.minimum(np.ones(num_boxes) * image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = np.minimum(np.ones(num_boxes) * image_size[0], np.floor(right + 0.5).astype('int32'))

    stacked = np.stack((top, left, bottom, right), axis=-1)
    return stacked