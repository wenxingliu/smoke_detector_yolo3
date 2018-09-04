import numpy as np
from scipy.spatial.distance import cdist, euclidean

from PIL import ImageFont, ImageDraw

__author__ = 'sliu'


def track_bboxes_between_two_adjacent_frames(bboxes_0, bboxes_1,
                                             image_size_0, image_size_1,
                                             top_N=5, distance_threshold=50):
    '''
    :param bboxes_0: list of bboxes in the previous frame
    :param bboxes_1: list of bboxes in the latter frame
    :param image_size_0: image size of the yolo output previous frame
    :param image_size_1: image size of the yolo output latter frame
    :param top_N: observe only top N bboxes in the previous frame
    :param distance_threshold: fo not connect dots if two bboxes centerpoints in prev/latter frame exceeds this value
    :return: paired bboxes coords
    '''
    # force negative coords to 0
    bboxes_0 = non_zero_coord_suppress_bboxes(bboxes_0, image_size_0)
    bboxes_1 = non_zero_coord_suppress_bboxes(bboxes_1, image_size_1)

    # show only top N biggest boxes
    filtered_bboxes_0 = filter_out_smaller_bboxes(bboxes_0, top_N=top_N)

    # find center points of each bbox and find the closest one in the next frame
    box_centerpoints_0 = compute_bboxes_centerpoints(filtered_bboxes_0)
    box_centerpoints_1 = compute_bboxes_centerpoints(bboxes_1)

    num_boxes_0 = len(filtered_bboxes_0)
    num_boxes_1 = len(bboxes_1)
    num_boxes_tracking = min(num_boxes_0, num_boxes_1)

    dist_mx = cdist(box_centerpoints_0, box_centerpoints_1)
    closest_bbox_indices = np.argmin(dist_mx, axis=1)

    paired_bboxes = np.zeros([num_boxes_tracking, 2, 4])
    tracked_boxes = 0

    for i, center_0 in enumerate(box_centerpoints_0):

        if i >= num_boxes_tracking:
            break

        mapped_index = closest_bbox_indices[i]
        center_1 = box_centerpoints_1[mapped_index]

        if not pair_should_be_filtered_out(center_0=center_0, center_1=center_1, distance_threshold=distance_threshold):
            paired_bboxes[tracked_boxes, 0,] = filtered_bboxes_0[i]
            paired_bboxes[tracked_boxes, 1,] = bboxes_1[mapped_index]
            tracked_boxes += 1

    return paired_bboxes[:tracked_boxes]


def non_zero_coord_suppress_bboxes(bboxes, image_size):
    num_boxes = len(bboxes)
    top, left, bottom, right = np.rollaxis(bboxes, -1)
    top = np.maximum(np.zeros(num_boxes), np.floor(top + 0.5).astype('int32'))
    left = np.maximum(np.zeros(num_boxes), np.floor(left + 0.5).astype('int32'))

    bottom = np.minimum(np.ones(num_boxes) * image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = np.minimum(np.ones(num_boxes) * image_size[0], np.floor(right + 0.5).astype('int32'))

    stacked = np.stack((top, left, bottom, right), axis=-1)
    return stacked


def compute_bboxes_centerpoints(bboxes):
    center_points = np.array([(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2]).T
    return center_points


def compute_bbox_sizes(bboxes):
    box_sizes = np.product([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]], axis=0)
    return box_sizes


def pair_should_be_filtered_out(center_0, center_1, distance_threshold, tol=0.01):
    # coming towards camera
    if center_0[0] < (center_1[0]) * (1 - tol):
        return True
    # centerpoints in two frames too far apart
    distance = euclidean(center_0, center_1)
    if distance > distance_threshold:
        return True
    # TODO: add crieria IOU too small
    return False


def filter_out_smaller_bboxes(bboxes, top_N=2):
    box_sizes = compute_bbox_sizes(bboxes)
    big_boxes_index = np.argsort(box_sizes)[::-1][:top_N]
    filtered_bboxes = bboxes[big_boxes_index]
    return filtered_bboxes


def plot_tracking_target(target_bboxes, images_dict, colors, every_n_frames=10):
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
        selected_img.show()


def track_biggest_bbox_in_first_image_for_n_frames(paired_bboxes_list, start_index=0, num_frames=25):
    track_target = {}

    paired_bboxes = paired_bboxes_list[start_index]
    bboxes_0, bboxes_1 = np.rollaxis(paired_bboxes, axis=1)
    target_index = find_biggest_bbox_index(bboxes_0)
    track_target[start_index] = bboxes_0[target_index]
    track_target[start_index + 1] = bboxes_1[target_index]

    for i, paired_bboxes in enumerate(paired_bboxes_list[start_index:(start_index + num_frames)]):
        if i == 0:
            continue

        bboxes_0, bboxes_1 = np.rollaxis(paired_bboxes, axis=1)
        index_in_bboxes_0 = find_index_of_given_bbox(track_target[start_index + i], bboxes_0)

        if index_in_bboxes_0 is None:
            break

        track_target[start_index + i + 1] = bboxes_1[target_index]

    return track_target


def compute_pairwise_bbox_match(yolo_detection_output_list, top_N):
    number_of_frames = len(yolo_detection_output_list)
    paired_bboxes_list = []
    for i in range(number_of_frames - 1):
        bboxes_0 = yolo_detection_output_list[i]['bboxes']
        bboxes_1 = yolo_detection_output_list[i+1]['bboxes']
        image_size_0 = yolo_detection_output_list[i]['image_size']
        image_size_1 = yolo_detection_output_list[i+1]['image_size']
        paired_bboxes = track_bboxes_between_two_adjacent_frames(bboxes_0, bboxes_1, image_size_0, image_size_1, top_N=top_N)
        paired_bboxes_list.append(paired_bboxes)
    return paired_bboxes_list


def find_biggest_bbox_index(bboxes):
    box_sizes = compute_bbox_sizes(bboxes)
    biggest_index = np.argmax(box_sizes)
    return biggest_index


def find_index_of_given_bbox(given_bbox, list_of_bboxes):
    for i, bbox in enumerate(list_of_bboxes):
        if (bbox == given_bbox).all():
             return i