import numpy as np
from scipy.spatial.distance import euclidean

__author__ = 'sliu'


def compute_bbox_sizes(bboxes):
    box_sizes = np.product([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=0)
    return box_sizes


def filter_small_bboxes(image_size, bboxes, ratio_threshold):
    bbox_sizes = compute_bbox_sizes(bboxes)
    whole_image_size = image_size[0] * image_size[1]
    ratios = bbox_sizes / whole_image_size
    filtered_bboxes = bboxes[ratios >= ratio_threshold]
    return filtered_bboxes

def filter_imbalanced_bboxes(bboxes,ratio_w_and_h):
    bbox_height, bbox_weight = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
    ratios = np.min((bbox_height / bbox_weight, bbox_weight / bbox_height), axis=0)
    filtered_bboxes = bboxes[ratios >= ratio_w_and_h]
    return filtered_bboxes

def bboxes_pair_should_be_filtered_out(bbox_0, bbox_1, center_0, center_1, distance_threshold, iou_threshold):
    # IOU too small
    if paired_boxes_iou_too_small(bbox_0, bbox_1, iou_threshold):
        return True

    distance = euclidean(center_0, center_1)
    if distance > distance_threshold:
        return True

    return False


def paired_boxes_iou_too_small(bbox_0, bbox_1, iou_threshold=0.9):
    iou = bboxes_intersection_over_union(bbox_0, bbox_1)
    return iou < iou_threshold


def bboxes_intersection_over_union(bbox_0, bbox_1):
    inter_area_top = max(bbox_0[0], bbox_1[0])
    inter_area_left = max(bbox_0[1], bbox_1[1])
    inter_area_bottom = min(bbox_0[2], bbox_1[2])
    inter_area_right = min(bbox_0[3], bbox_1[3])
    inter_bbox = np.array([inter_area_top, inter_area_left, inter_area_bottom, inter_area_right])
    bbox_size_0, bbox_size_1, inter_bbox_size = compute_bbox_sizes(np.array([bbox_0, bbox_1, inter_bbox]))
    iou = inter_bbox_size / float(bbox_size_0 + bbox_size_1 - inter_bbox_size)
    return iou


def aug_bbox_range(bbox, image_size, w_aug_factor, h_aug_factor):
    top, left, bottom, right = bbox
    h, w = (bottom - top), (right - left)
    # crop_top = np.max([0, top - h*h_aug_factor]).astype(int)
    crop_top = np.max([0, top]).astype(int)
    # crop_left = np.max([0, left - w*w_aug_factor]).astype(int)
    crop_left = np.max([0, left - w*w_aug_factor]).astype(int)
    crop_bottom = np.min([image_size[1], bottom + h*h_aug_factor]).astype(int)
    crop_right = np.min([image_size[0], right + w*w_aug_factor]).astype(int)
    return crop_top, crop_left, crop_bottom, crop_right