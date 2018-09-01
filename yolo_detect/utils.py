import numpy as np
from models.utils import box_iou

def filter_small_objects(boxes, image_size, area_threshold=0.1):
    box_areas = np.product([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=0)
    image_area = np.product(image_size)
    return [i for i in np.arange(len(box_areas)) if box_areas[i] >= image_area * area_threshold]

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
