import os
from PIL import Image
from tf_yolov3.yolo3_model_tf import *
from tf_yolov3.utils import letterbox_image


class YOLO(object):
    def __init__(self):
        self.default_path = {
            "model_path": 'yolo_weights/...',  # tf_weights of yolo3
            "anchors_path": 'model_data/yolo_anchors.txt',
            "classes_path": 'model_data/coco_classes.txt',
            "vehicle_classes_path": 'model_data/vehicle_classes.txt',
        }
        self.anchors = self.get_anchors()
        self.vehicle_class_names = self.get_vehicle_class()
        self.iou = 0.5
        self.score = 0.5
        self.input_image_size = (416, 416)
        self.test_batch = 1
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.vehicle_class_names)
        self.sess = tf.Session()

    def get_class(self):
        classes_path = os.path.expanduser(self.default_path["classes_path"])
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_vehicle_class(self):
        vehicle_classes_path = os.path.expanduser(self.default_path["vehicle_classes_path"])
        with open(vehicle_classes_path) as f:
            vehicle_classes_names = f.readlines()
        vehicle_classes_names = [c.strip() for c in vehicle_classes_names]
        return vehicle_classes_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.default_path["anchors_path"])
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def yolo_body(self, inputs, num_anchors, num_classes):
        x1, x2, x3 = darknet52(inputs)
        x, y1 = make_lastlayer(x1, 512, num_anchors * (num_classes + 5))

        x = conv2d(x, 256, (1, 1), strides=[1, 1])
        x = upsample2d(x, 2, 2)
        x = tf.concat([x, x2], -1)
        x, y2 = make_lastlayer(x, 256, num_anchors * (num_classes + 5))

        x = conv2d(x, 128, (1, 1), strides=[1, 1])
        x = upsample2d(x, 2, 2)
        x = tf.concat([x, x3], -1)
        x, y3 = make_lastlayer(x, 128, num_anchors * (num_classes + 5))

        return [y1, y2, y3]

    def yolo_evaluate(self, yolo_outputs, anchors, num_classes,
                      image_shape, max_boxes=20, score_threshold=0.6, iou_threshold=0.4):
        num_layers = len(yolo_outputs)
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        # three feature map of different scale to detect
        for i in range(num_layers):
            anchor = anchors[anchors_mask[i]]
            _box, _box_score = yolo_boxes_and_scores(yolo_outputs[i], anchor, num_classes, input_shape, image_shape)
            boxes.append(_box)
            box_scores.append(_box_score)
        boxes = tf.concat(boxes, axis=0)  # shape:[n, 4]
        box_scores = tf.concat(box_scores, axis=0)  # shape:[n, 80]

        mask = box_scores >= score_threshold  # shape: [n, 80]
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

    def yolo_model(self, inputs_data, inputs_image_size):
        # inference
        yolo_output = self.yolo_body(inputs_data, self.num_anchors, self.num_classes)
        out_boxes, out_scores, out_classes = self.yolo_evaluate(yolo_output, self.anchors, self.num_classes,
                                                                inputs_image_size, score_threshold=self.score,
                                                                iou_threshold=self.iou)
        return out_boxes, out_scores, out_classes

    def make_input_placeholders(self):
        # define input placeholder
        inputs_data = tf.placeholder(tf.float32, [self.test_batch, 416, 416, 3])
        inputs_image_size = tf.placeholder(tf.int8, [2, ])
        return inputs_data, inputs_image_size

    def detect_image(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.input_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        image_size = [image.size[1], image.size[0]]

        inputs_data, inputs_image_size = self.make_input_placeholders()
        boxes_, scores_, classes_ = self.yolo_model(inputs_data, inputs_image_size)
        self.sess.run(tf.global_variables_initializer())
        out_boxes, out_scores, out_classes = \
            self.sess.run([boxes_, scores_, classes_],
                          feed_dict={inputs_data: image_data, inputs_image_size: image_size})
        self.close_setion()
        return out_boxes, out_scores, out_classes

    def close_setion(self):
        self.sess.close()


yolo3 = YOLO()
image = Image.open('2.png')

out_boxes, out_scores, out_classes = yolo3.detect_image(image)












