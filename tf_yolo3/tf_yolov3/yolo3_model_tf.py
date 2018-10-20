import tensorflow as tf
import numpy as np


def conv2d(inputs, filters, ksize, strides=[1, 1]):
    conv = tf.layers.conv2d(inputs, filters, ksize, strides, padding="valid" if strides == [2, 2] else "same")
    bn = tf.layers.batch_normalization(conv)
    layer = tf.nn.leaky_relu(bn, alpha=0.01)
    return layer


def Res_block(inputs, shortcut_inputs):
    layer = tf.add(inputs, shortcut_inputs)
    layer = tf.nn.leaky_relu(layer, alpha=0.01)
    return layer


def darknet52(inputs):

    layer1 = conv2d(inputs, 32, (3, 3), strides=[1, 1])
    layer2 = conv2d(layer1, 64, (3, 3), strides=[2, 2])

    # resblock1
    layer3 = conv2d(layer2, 32, (1, 1), strides=[1, 1])
    layer4 = conv2d(layer3, 64, (3, 3), strides=[1, 1])
    resblock1 = Res_block(layer4, layer2)

    layer5 = conv2d(resblock1, 128, (3, 3), strides=[2, 2])
    
    # resblock2 * 2
    layer6 = conv2d(layer5,  64, (1, 1), strides=[1, 1])
    layer7 = conv2d(layer6, 128, (3, 3), strides=[1, 1])
    resblock2 = Res_block(layer7, layer5)

    layer8 = conv2d(resblock2, 64, (1, 1), strides=[1, 1])
    layer9 = conv2d(layer8, 128, (3, 3), strides=[1, 1])
    resblock3 = Res_block(layer9, resblock2)

    layer10 = conv2d(resblock3, 256, (3, 3), strides=[2, 2])
   
    # resblock3 * 8
    layer11 = conv2d(layer10, 128, (1, 1), strides=[1, 1])
    layer12 = conv2d(layer11, 256, (3, 3), strides=[1, 1])
    resblock4 = Res_block(layer12, layer10)

    layer13 = conv2d(resblock4, 128, (1, 1), strides=[1, 1])
    layer14 = conv2d(layer13, 256, (3, 3), strides=[1, 1])
    resblock5 = Res_block(layer14, resblock4)

    layer15 = conv2d(resblock5, 128, (1, 1), strides=[1, 1])
    layer16 = conv2d(layer15, 256, (3, 3), strides=[1, 1])
    resblock6 = Res_block(layer16, resblock5)

    layer17 = conv2d(resblock6, 128, (1, 1), strides=[1, 1])
    layer18 = conv2d(layer17, 256, (3, 3), strides=[1, 1])
    resblock7 = Res_block(layer18, resblock6)

    layer19 = conv2d(resblock7, 128, (1, 1), strides=[1, 1])
    layer20 = conv2d(layer19, 256, (3, 3), strides=[1, 1])
    resblock8 = Res_block(layer20, resblock7)

    layer21 = conv2d(resblock8, 128, (1, 1), strides=[1, 1])
    layer22 = conv2d(layer21, 256, (3, 3), strides=[1, 1])
    resblock9 = Res_block(layer22, resblock8)

    layer23 = conv2d(resblock9, 128, (1, 1), strides=[1, 1])
    layer24 = conv2d(layer23, 256, (3, 3), strides=[1, 1])
    resblock10 = Res_block(layer24, resblock9)

    layer25 = conv2d(resblock10, 128, (1, 1), strides=[1, 1])
    layer26 = conv2d(layer25, 256, (3, 3), strides=[1, 1])
    resblock11 = Res_block(layer26, resblock10)             # (52,52)
    
    layer27 = conv2d(resblock11, 512, (3, 3), strides=[2, 2])
    
    # resblock4 * 8
    layer28 = conv2d(layer27, 256, (1, 1), strides=[1, 1])
    layer29 = conv2d(layer28, 512, (3, 3), strides=[1, 1])
    resblock12 = Res_block(layer29, layer27)

    layer30 = conv2d(resblock12, 256, (1, 1), strides=[1, 1])
    layer31 = conv2d(layer30, 512, (3, 3), strides=[1, 1])
    resblock13 = Res_block(layer31, resblock12)

    layer32 = conv2d(resblock13, 256, (1, 1), strides=[1, 1])
    layer33 = conv2d(layer32, 512, (3, 3), strides=[1, 1])
    resblock14 = Res_block(layer33, resblock13)

    layer34 = conv2d(resblock14, 256, (1, 1), strides=[1, 1])
    layer35 = conv2d(layer34, 512, (3, 3), strides=[1, 1])
    resblock15 = Res_block(layer35, resblock14)

    layer36 = conv2d(resblock15, 256, (1, 1), strides=[1, 1])
    layer37 = conv2d(layer36, 512, (3, 3), strides=[1, 1])
    resblock16 = Res_block(layer37, resblock15)

    layer38 = conv2d(resblock16, 256, (1, 1), strides=[1, 1])
    layer39 = conv2d(layer38, 512, (3, 3), strides=[1, 1])
    resblock17 = Res_block(layer39, resblock16)

    layer40 = conv2d(resblock17, 256, (1, 1), strides=[1, 1])
    layer41 = conv2d(layer40, 512, (3, 3), strides=[1, 1])
    resblock18 = Res_block(layer41, resblock17)

    layer42 = conv2d(resblock18, 256, (1, 1), strides=[1, 1])
    layer43 = conv2d(layer42, 512, (3, 3), strides=[1, 1])
    resblock19 = Res_block(layer43, resblock18)             # (26, 26)

    layer44 = conv2d(resblock19, 1024, (3, 3), strides=[2, 2])

    # resblock5 *4
    layer45 = conv2d(layer27, 512, (1, 1), strides=[1, 1])
    layer46 = conv2d(layer28, 1024, (3, 3), strides=[1, 1])
    resblock20 = Res_block(layer46, layer44)

    layer47 = conv2d(resblock20, 128, (1, 1), strides=[1, 1])
    layer48 = conv2d(layer47, 256, (3, 3), strides=[1, 1])
    resblock21 = Res_block(layer48, resblock20)

    layer49 = conv2d(resblock21, 128, (1, 1), strides=[1, 1])
    layer50 = conv2d(layer49, 256, (3, 3), strides=[1, 1])
    resblock22 = Res_block(layer50, resblock21)

    layer51 = conv2d(resblock22, 128, (1, 1), strides=[1, 1])
    layer52 = conv2d(layer51, 256, (3, 3), strides=[1, 1])
    resblock23 = Res_block(layer52, resblock22)         # (13,13)

    return resblock11, resblock19, resblock23


def make_lastlayer(inputs, filters, out_filters):

    layer = conv2d(inputs, filters, (1, 1))
    layer = conv2d(layer, filters*2, (3, 3))
    layer = conv2d(layer, filters, (1, 1))
    layer = conv2d(layer, filters*2, (3, 3))
    layer = conv2d(layer, filters, (1, 1))

    layer2 = conv2d(layer, filters*2, (3, 3))
    layer_y = tf.layers.conv2d(layer2, out_filters, (1, 1))

    return layer, layer_y


def upsample2d(inputs, height_factor, width_factor):

    original_shape = tf.shape(inputs)[1:3]
    new_shape = original_shape * tf.constant(np.array([height_factor, width_factor]).astype('int32'))
    x = tf.image.resize_nearest_neighbor(inputs, new_shape)
    x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                 original_shape[2] * width_factor if original_shape[2] is not None else None, None))
    return x


def yolo_body(inputs, num_anchors, num_classes):

    x1, x2, x3 = darknet52(inputs)
    x, y1 = make_lastlayer(x1, 512, num_anchors * (num_classes + 5))

    x = conv2d(x, 256, (1, 1), strides=[1, 1])
    x = upsample2d(x)
    x = tf.concat([x, x1])
    x, y2 = make_lastlayer(x, 256, num_anchors * (num_classes + 5))

    x = conv2d(x, 128, (1, 1), strides=[1, 1])
    x = upsample2d(x)
    x = tf.concat([x, x2])
    x, y3 = make_lastlayer(x, 128, num_anchors * (num_classes + 5))

    return [y1, y2, y3]


def yolo_head(final_layer, anchors, num_classes, input_shape, calculate_loss=True):

    num_anchors = len(anchors)
    anchors_tensor = tf.reshape(tf.constant(anchors, shape=[1, 1, 1, num_anchors, 2]))
    grid_shape = tf.shape(final_layer)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[0], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[1], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y])
    grid = tf.cast(grid, dtype=final_layer.dtype.base_dtype.name)

    feats = tf.reshape(final_layer, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # feats[..., :2] 表示取前边所有维度和最后一维度上的前两个变量（最后一个维度上有85个变量，前四个是bx，by，bw，bh）
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype.base_dtype.name)  # 中心点坐标预测
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype.base_dtype.name)  # 宽高预测
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calculate_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correctboxes(box_xy, box_wh, input_shape, image_shape):

    """input_shape: (416,416)
    image_shape: raw image size, maybe (1920,1080) or others"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype.base_dtype.name)
    image_shape = tf.cast(image_shape, box_yx.dtype.base_dtype.name)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))  # make max(h,w)==input shape
    offset = (input_shape - new_shape)/2.0/input_shape
    # box_yx center points position in raw image(raw image size=new_shape)
    box_yx = (box_yx - offset) * (input_shape/new_shape)
    box_hw = box_hw * (input_shape/new_shape)
    # box_min, box_max is the four corners' coordinates in the raw image(raw image size=new_shape)
    box_min = box_yx - (box_hw/2.0)
    box_max = box_yx + (box_hw/2.0)
    boxes = tf.concat([box_min[..., 0], box_min[..., 1], box_max[..., 0], box_max[..., 1]], axis=-1)
    boxes = boxes * tf.concat([image_shape, image_shape], axis=-1)

    return boxes


def yolo_boxes_and_scores(final_layer, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(final_layer, anchors, num_classes,
                                                                input_shape, calculate_loss=False)
    boxes = yolo_correctboxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_evaluate(yolo_outputs, anchors, num_classes,
                  image_shape, max_boxes=20, score_threshold=0.6, iou_threshold=0.4):
    num_layers = len(yolo_outputs)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = tf.shape(yolo_outputs[1:3]) * 32
    boxes = []
    box_scores = []
    # three feature map of different scale to detect
    for i in range(num_layers):
        anchors = anchors[anchors_mask[i]]
        _box, _box_score = yolo_boxes_and_scores(yolo_outputs[i], anchors, num_classes, input_shape, image_shape)
        boxes.append(_box)
        box_scores.append(_box_score)
    boxes = tf.concat(boxes, axis=0)    # shape:[n, 4]
    box_scores = tf.concat(box_scores, axis=0)  # shape:[n, 80]

    mask = box_scores >= score_threshold    # shape: [n, 80]
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold)
        class_boxes = tf.nn.embedding_lookup(class_boxes, nms_index)
        class_box_scores = tf.nn.embedding_lookup(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    # bbox size and location, bbox scores, bbox classes  has same first dims .
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)
    return boxes_, scores_, classes_


def pre_process_true_box():
    return None


def yolo_loss():
    return None