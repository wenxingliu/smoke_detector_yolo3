import numpy as np

def sort_boxes_by_confidence_and_size(out_boxes, out_scores, score_threshold):
    box_area = np.product(out_boxes[:, 2:4], axis=1)
    high_confidence_boxes_indices = np.where(out_scores > score_threshold)[0]
    sorted_box_area_index = box_area.argsort()[::-1]
    sorted_indices = np.array([i for i in sorted_box_area_index if i in high_confidence_boxes_indices])
    return sorted_indices
