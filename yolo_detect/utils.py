import json
import numpy as np
from models.utils import box_iou

def filter_small_objects(boxes, image_size, area_threshold=0.1):
    box_areas = np.product([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=0)
    image_area = np.product(image_size)
    return [i for i in np.arange(len(box_areas)) if box_areas[i] >= image_area * area_threshold]


def find_biggest_bbox_index(boxes):
    box_areas = np.product([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=0)
    max_bbox_index = np.argmax(box_areas)
    return max_bbox_index


def is_bbox_in_center(top, left, bottom, right, image_size, v_region=(0.5, 0.75), h_region=(0, 1)):
    center_coords = find_bbox_center(bottom, left, right, top)
    if (center_coords[1] < image_size[1] * np.min(v_region)) | (center_coords[1] > image_size[1] * np.max(v_region)):
        return False

    if (center_coords[0] < image_size[0] * np.min(h_region)) | (center_coords[0] > image_size[0] * np.max(h_region)):
        return False

    return True


def discard_overlapping_boxes(boxes, overlap_iou_threshold=0.8):
    delete_indices = []
    indices = np.arange(len(boxes))

    for i in indices[:-1]:
        for j in indices[(i+1):]:
            box_i = boxes[i]
            box_j = boxes[j]
            iou = box_iou(box_i, box_j)
            if iou >= overlap_iou_threshold:
                delete_indices.append(j)

    return np.array(delete_indices)


def log_detection_outputs_to_json(bboxes, scores, labels, image_size, vehicle_box_indices, output_filename=None):
    json_file = {"bboxes": bboxes,
                 "scores": scores,
                 "labels": labels,
                 "image_size": image_size,
                 "vehicle_box_indices": vehicle_box_indices}
    if output_filename is None:
        return json_file
    with open(output_filename, 'w') as outfile:
        json.dump(json_file, outfile)


def find_bbox_corners(box, image_size):
    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image_size[0], np.floor(right + 0.5).astype('int32'))
    return top, left, bottom, right

def find_bbox_center(bottom, left, right, top):
    return [(right + left) / 2, (top + bottom) / 2]

# vectorized
def compute_bboxes_centerpoints(bboxes):
    num_boxes = len(bboxes)
    top, left, bottom, right = np.rollaxis(bboxes, -1)
    top = np.maximum(np.zeros(num_boxes), top)
    left = np.maximum(np.zeros(num_boxes), left)
    bottom = np.maximum(np.zeros(num_boxes), bottom)
    right = np.maximum(np.zeros(num_boxes), right)
    center_points = np.array([(top + bottom)/2, (right + left)/2]).T
    return center_points