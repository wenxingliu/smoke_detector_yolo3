import numpy as np
from scipy.spatial.distance import cdist

from log_utils import save_numpy_file, save_image_to_file
from track_vehicle.utils import is_bbox_leaving_camera, compute_bbox_sizes, find_index_of_given_bbox, \
    find_bigger_bboxes_indices, pair_should_be_filtered_out, aug_bbox_range

__author__ = 'sliu'


def bounding_box_tracking(yolo_outputs, output_dir, top_N=5, track_num_frames=20):
    paired_bboxes_list = compute_pairwise_bbox_match(yolo_outputs, top_N=top_N)
    tracked_bboxes_dict = track_all_bboxes_for_n_frames(paired_bboxes_list,
                                                        start_frame_index=0,
                                                        look_at_num_frames=len(yolo_outputs),
                                                        track_num_frames=track_num_frames)

    # save to file
    save_numpy_file(output_dir, 'tracked_bboxes_dict', np.array([tracked_bboxes_dict]))

    return tracked_bboxes_dict


def crop_bbox_from_image(yolo_outputs, images_dict, tracked_bboxes_dict, output_dir,
                         w_aug_factor=0.2, h_aug_factor=0.2, interval=10, export_every_n=5,
                         save_cropped_image=False):
    cropped_images = {}
    previous_index = -1 * interval

    for i, track_path in tracked_bboxes_dict.items():
        if i - previous_index <= interval:
            continue

        previous_index = i

        for j, bbox in track_path.items():
            if (i != j) and (j - i) % export_every_n == 0:
                processed_image = np.array(images_dict[j], dtype='float32')
                image_size = yolo_outputs[j]['image_size']
                crop_top, crop_left, crop_bottom, crop_right = aug_bbox_range(bbox, image_size,
                                                                              w_aug_factor, h_aug_factor)
                cropped_img = processed_image[crop_top:crop_bottom, crop_left:crop_right, ]
                cropped_images[(i, j)] = cropped_img

                if save_cropped_image:
                    image_file_name = 'cropped_%d_%d' % (i, j)
                    save_image_to_file(output_dir, image_file_name, cropped_img)


    # save to file
    save_numpy_file(output_dir, 'cropped_images', np.array([cropped_images]))

    return cropped_images


def track_all_bboxes_for_n_frames(paired_bboxes_list, start_frame_index, look_at_num_frames,
                                  track_num_frames):
    tracked_boxes_dict = {}
    track_box_size_dict = {}

    for frame_no in range(look_at_num_frames - 1):
        frame_index = frame_no + start_frame_index

        paired_bboxes = paired_bboxes_list[frame_index]

        for target_index in range(paired_bboxes.shape[0]):
            track_path = track_a_bbox_for_n_frames(paired_bboxes_list, frame_index, target_index, track_num_frames)

            if is_bbox_leaving_camera(np.array(list(track_path.values()))):

                current_box_size = compute_bbox_sizes(np.array([track_path[frame_index]]))

                # TODO: finds the box in the middle
                if (
                        frame_index in tracked_boxes_dict
                        and tracked_boxes_dict[frame_index][frame_index][2] < track_path[frame_index][2]
                        and current_box_size < track_box_size_dict[frame_index]
                        and len(track_path) >= track_num_frames
                ):
                    continue

                tracked_boxes_dict[frame_index] = track_path
                track_box_size_dict[frame_index] = current_box_size

    return tracked_boxes_dict


def track_a_bbox_for_n_frames(paired_bboxes_list, start_index=0, target_index=0, num_frames=25):
    track_target = {}

    paired_bboxes = paired_bboxes_list[start_index]

    if len(paired_bboxes) == 0:
        return track_target

    bboxes_0, bboxes_1 = np.rollaxis(paired_bboxes, axis=1)

    track_target[start_index] = bboxes_0[target_index]
    track_target[start_index + 1] = bboxes_1[target_index]

    for i, paired_bboxes in enumerate(paired_bboxes_list[start_index:(start_index + num_frames)]):
        if i == 0:
            continue

        bboxes_0, bboxes_1 = np.rollaxis(paired_bboxes, axis=1)
        index_in_bboxes_0 = find_index_of_given_bbox(track_target[start_index + i], bboxes_0)

        if index_in_bboxes_0 is None:
            break

        track_target[start_index + i + 1] = bboxes_1[index_in_bboxes_0]

    return track_target


def track_bboxes_between_two_adjacent_frames(bboxes_0, bboxes_1,
                                             box_centerpoints_0, box_centerpoints_1,
                                             top_N=5, distance_threshold=50, iou_threshold=0.8):
    '''
    :param bboxes_0: list of bboxes in the previous frame
    :param bboxes_1: list of bboxes in the latter frame
    :param image_size_0: image size of the yolo output previous frame
    :param image_size_1: image size of the yolo output latter frame
    :param top_N: observe only top N bboxes in the previous frame
    :param distance_threshold: fo not connect dots if two bboxes centerpoints in prev/latter frame exceeds this value
    :return: paired bboxes coords
    '''

    # show only top N biggest boxes
    filtered_bboxes_indices_0 = find_bigger_bboxes_indices(bboxes_0, top_N=top_N)
    filtered_bboxes_0 = bboxes_0[filtered_bboxes_indices_0]
    filtered_box_centerpoints_0 = box_centerpoints_0[filtered_bboxes_indices_0]

    num_boxes_0 = len(filtered_bboxes_0)
    num_boxes_1 = len(bboxes_1)
    num_boxes_tracking = min(num_boxes_0, num_boxes_1)

    dist_mx = cdist(filtered_box_centerpoints_0, box_centerpoints_1)
    closest_bbox_indices = np.argmin(dist_mx, axis=1)

    paired_bboxes = np.zeros([num_boxes_tracking, 2, 4])
    tracked_boxes = 0

    for i, center_0 in enumerate(filtered_box_centerpoints_0):

        if i >= num_boxes_tracking:
            break

        mapped_index = closest_bbox_indices[i]
        center_1 = box_centerpoints_1[mapped_index]

        bbox_0 = filtered_bboxes_0[i]
        bbox_1 = bboxes_1[mapped_index]

        if not pair_should_be_filtered_out(bbox_0=bbox_0, bbox_1=bbox_1,
                                           center_0=center_0, center_1=center_1,
                                           distance_threshold=distance_threshold,
                                           iou_threshold=iou_threshold):
            paired_bboxes[tracked_boxes, 0,] = bbox_0
            paired_bboxes[tracked_boxes, 1,] = bbox_1
            tracked_boxes += 1

    return paired_bboxes[:tracked_boxes]


def compute_pairwise_bbox_match(yolo_detection_output_list, top_N):
    number_of_frames = len(yolo_detection_output_list)
    paired_bboxes_list = []
    for i in range(number_of_frames - 1):
        bboxes_0 = yolo_detection_output_list[i]['bboxes']
        bboxes_1 = yolo_detection_output_list[i+1]['bboxes']
        box_centerpoints_0 = yolo_detection_output_list[i]['centerpoints']
        box_centerpoints_1 = yolo_detection_output_list[i+1]['centerpoints']
        paired_bboxes = track_bboxes_between_two_adjacent_frames(bboxes_0, bboxes_1,
                                                                 box_centerpoints_0,
                                                                 box_centerpoints_1,
                                                                 top_N=top_N)
        paired_bboxes_list.append(paired_bboxes)
    return paired_bboxes_list