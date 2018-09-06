import cv2
import colorsys

import numpy as np
from PIL import ImageDraw
from scipy.spatial.distance import euclidean

from yolo_detect.utils import compute_bboxes_centerpoints


__author__ = 'sliu'


def compute_bbox_sizes(bboxes):
    box_sizes = np.product([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=0)
    return box_sizes


def pair_should_be_filtered_out(bbox_0, bbox_1, center_0, center_1, distance_threshold, iou_threshold):
    # IOU too small
    if paired_boxes_iou_too_small(bbox_0, bbox_1, iou_threshold):
        return True

    distance = euclidean(center_0, center_1)
    if distance > distance_threshold:
        return True
    return False


def is_bbox_leaving_camera(tracked_bboxes, threshold=0.8):
    if len(tracked_bboxes) < 10:
        return False
    centerpoints = compute_bboxes_centerpoints(tracked_bboxes)
    moving_further_rate = np.mean([centerpoints[i,1] - centerpoints[i+1,1] > 0 for i in range(centerpoints.shape[0] - 1)])
    return moving_further_rate >= threshold


def paired_boxes_iou_too_small(bbox_0, bbox_1, iou_threshold=0.9):
    iou = bboxes_intersection_over_union(bbox_0, bbox_1)
    return iou < iou_threshold


def paired_boxes_iou_too_large(bbox_0, bbox_1, iou_threshold=0.9):
    iou = bboxes_intersection_over_union(bbox_0, bbox_1)
    return iou >= iou_threshold


def bboxes_intersection_over_union(bbox_0, bbox_1):
    inter_area_top = max(bbox_0[0], bbox_1[0])
    inter_area_left = max(bbox_0[1], bbox_1[1])
    inter_area_bottom = min(bbox_0[2], bbox_1[2])
    inter_area_right = min(bbox_0[3], bbox_1[3])
    inter_bbox = np.array([inter_area_top, inter_area_left, inter_area_bottom, inter_area_right])
    bbox_size_0, bbox_size_1, inter_bbox_size = compute_bbox_sizes(np.array([bbox_0, bbox_1, inter_bbox]))
    iou = inter_bbox_size / float(bbox_size_0 + bbox_size_1 - inter_bbox_size)
    return iou


def find_bigger_bboxes_indices(bboxes, top_N):
    box_sizes = compute_bbox_sizes(bboxes)
    big_boxes_index = np.argsort(box_sizes)[::-1][:top_N]
    return big_boxes_index


def find_index_of_given_bbox(given_bbox, list_of_bboxes):
    for i, bbox in enumerate(list_of_bboxes):
        if (bbox == given_bbox).all():
             return i


def generate_colors_dict(N):
    hsv_tuples = [(x / 13, 1., 1.) for x in range(N)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def plot_tracking_target(target_bboxes, images_dict, colors, every_n_frames=10, output_file_path=None):
    color_index = np.random.randint(0, len(colors))

    for i, box in target_bboxes.items():
        if i % every_n_frames != 0:
            continue

        selected_img = images_dict[i]
        thickness = (selected_img.size[0] + selected_img.size[1]) // 300

        draw = ImageDraw.Draw(selected_img)
        top, left, bottom, right = box

        for l in range(thickness):
            draw.rectangle([left + l, top + l, right - l, bottom - l], outline=colors[color_index])
        del draw

        if output_file_path is None:
            selected_img.show()
        else:
            output_file_name = output_file_path + str(i) + '.jpg'
            cv2.imwrite(output_file_name, np.array(selected_img, dtype='float32'))