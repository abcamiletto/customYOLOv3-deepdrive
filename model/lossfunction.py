#!/usr/bin/env python
# coding: utf-8


# Core libraries
import numpy as np
import tensorflow as tf
from utils import broadcast_iou, xywh_to_y1x1y2x2, xywh_to_x1x2y1y2
from datetime import datetime

# Anchors
yolo_anchors_tf = tf.constant([(10, 10), (22, 23), (47, 33), (39, 81), (82, 54), (127, 86),
                         (118, 168), (194, 130), (257, 221)], tf.float32)


def decode_output(y_pred, valid_anchors, num_classes = 10):
    t_xy, t_wh, objectness, classes = tf.split(y_pred, (2, 2, 1, num_classes), axis=-1)

    # That's because it's a Logistic classifier
    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)

    o_xy = tf.sigmoid(t_xy)
    b_wh = tf.exp(t_wh) * valid_anchors

    y_pred = tf.concat([o_xy, b_wh, objectness, classes], axis=-1)
    return y_pred

def from_rel_to_abs(y_pred, num_classes = 10):
    """
    Given a cell offset prediction from the model, calculate the absolute box coordinates to the whole image.
    note that, we divide w and h by grid size
    INPUTS:
    y_pred: Prediction tensor decoded from the model output, in the shape of (batch, grid, grid, anchor, 5 + num_classes)
    OUTPUTS:
    y_box: boxes in shape of (batch, grid, grid, anchor, 4), the last dimension is (x,y,w,h)
    objectness: probability that an object exists
    classes: probability of classes
    """

    o_xy, b_wh, objectness, classes = tf.split(y_pred, (2, 2, 1, num_classes), axis=-1)

    grid_size = tf.shape(y_pred)[1]

##########################

    # Now i'm gonna get a tensor like this
    #
    # [[[[0, 0]], [[1, 0]], [[2, 0]]],
    #  [[[0, 1]], [[1, 1]], [[2, 1]]],
    #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
    #
    # we have a grid, which can always give us (y, x)
    # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]

    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.stack(C_xy, axis=-1)
    C_xy = tf.expand_dims(C_xy, axis=2)

##########################

    # bx = sigmoid(tx) + Cx
    # by = sigmoid(ty) + Cy
    #
    # for example, if all elements in b_xy are (0.1, 0.2), the result will be
    #
    # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
    #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
    #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]

    b_xy = o_xy + tf.cast(C_xy, tf.float32)

    # finally, divide this absolute box_xy by grid_size, and then we will get the normalized bbox centroids
    # for each anchor in each grid cell. b_xy is now in shape (batch_size, grid_size, grid_size, num_anchor, 2)
    #
    # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
    #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
    #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
    #
    b_xy = b_xy / tf.cast(grid_size, tf.float32)

##########################

    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box , objectness, classes



def encode_output(y_true, valid_anchor):
    """
    This is the inverse of `decode_into_abs` above. It's turning (bx, by, bw, bh) into
    (tx, ty, tw, th) that is relative to cell location.
    """
    grid_size = tf.shape(y_true)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.expand_dims(tf.stack(C_xy, axis=-1), axis=2)

    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]
    t_xy = b_xy * tf.cast(grid_size, tf.float32) - tf.cast(C_xy, tf.float32)

    t_wh = tf.math.log(b_wh / valid_anchor)
    # b_wh could have some cells are 0, divided by anchor could result in inf or nan
    t_wh = tf.where(
        tf.logical_or(tf.math.is_inf(t_wh), tf.math.is_nan(t_wh)),
        tf.zeros_like(t_wh), t_wh)

    y_box = tf.concat([t_xy, t_wh], axis=-1)
    return y_box



def BinaryCrossentropy(pred_prob, labels):
    # I use a custom crossentropy because i need to weight diffently the 2 parts
    epsilon = 1e-7
    pred_prob = tf.clip_by_value(pred_prob, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(pred_prob) +
             (1 - labels) * tf.math.log(1 - pred_prob))

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, size, anchors = yolo_anchors_tf, **kwargs):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5
        self.anchors = anchors
        self.size = size
        if self.size == 'dense':
            self.valid_anchors = self.anchors[0:3]
        if self.size == 'medium':
            self.valid_anchors = self.anchors[3:6]
        if self.size == 'coarse':
            self.valid_anchors = self.anchors[6:9]
        self.writer = tf.summary.create_file_writer("/home/andrea/AI/ispr_yolo/NOTEBOOKS/training/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '/' + self.size)
        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight = None): # In order to call it like a function
        """
        calculate the loss of model prediction for one scale
        """
        # suffix rel (relative) means that its coordinates are relative to cells
        # basically (tx, ty, tw, th) format from the paper
        # _rel is used to calcuate the loss
        # suffix abs (absolute) means that its coordinates are absolute with in whole image
        # basically (bx, by, bw, bh) format from the paper
        # _abs is used to calcuate iou and ignore mask


######### sort of "Non Max Suppression" to get an ignore mask, we need the absolute values!

        # this box is used to calculate IoU, NOT loss.
        y_pred_decoded = decode_output(y_pred, self.valid_anchors)
        pred_box_abs, pred_obj, pred_class = from_rel_to_abs(y_pred_decoded, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)

        # split y_true into xy, wh, objectness and one-hot classes
        true_box_abs, true_obj, true_class = from_rel_to_abs(y_true, self.num_classes)
        y_true_abs = tf.concat([true_box_abs, true_obj, true_class], axis=-1)
        true_xy_abs = true_box_abs[..., 0:2]
        true_wh_abs = true_box_abs[..., 2:4]
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)

        # use the absolute yolo box to calculate iou and ignore mask
        ignore_mask = self.nonmax_mask(true_obj, true_box_abs,
                                            pred_box_abs)

######### Getting properly formatted values to process the loss functions

        # true_box_rel: (batch, grid, grid, anchor, 4)
        true_box_encoded = encode_output(y_true_abs, self.valid_anchors)
        true_xy_encoded = true_box_encoded[..., 0:2]
        true_wh_encoded = true_box_encoded[..., 2:4]

        # some adjustment to improve small box detection, note the (2-truth.w*truth.h) below
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        # YoloV2:
        # "If the cell is offset from the top left corner of the image by (cx , cy)
        # and the bounding box prior has width and height pw , ph , then the predictions correspond to:"
        #
        # to calculate the iou and determine the ignore mask, we need to first transform
        # prediction into real coordinates (bx, by, bw, bh)


        # split y_pred into xy, wh, objectness and one-hot classes
        # pred_??_rel: (batch, grid, grid, anchor, 2)
        pred_xy_encoded = y_pred[..., 0:2]
        pred_wh_encoded = y_pred[..., 2:4]

        # YoloV2:
        # "This ground truth value can be easily computed by inverting the equations above."
        #
        # to calculate loss and differentiation, we need to transform ground truth into
        # cell offset first
        xy_loss = self.calc_xy_loss(true_obj, true_xy_encoded, pred_xy_encoded, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_encoded, pred_wh_encoded, weight)

        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)
        
        
        tensorboard_names = ["batch xy loss " + self.size,
                             "batch wh loss " + self.size,
                             "batch class loss " + self.size,
                             "batch obj loss " + self.size]
                             
        with self.writer.as_default():
            tf.summary.scalar(tensorboard_names[0], tf.reshape(tf.reduce_sum(xy_loss), []), step=1)
            tf.summary.scalar(tensorboard_names[1], tf.reshape(tf.reduce_sum(wh_loss), []), step=1)
            tf.summary.scalar(tensorboard_names[2], tf.reshape(tf.reduce_sum(class_loss), []), step=1)
            tf.summary.scalar(tensorboard_names[3], tf.reshape(tf.reduce_sum(obj_loss), []), step=1)
            self.writer.flush()
        
        # YoloV1: Function (3)
        return xy_loss + wh_loss + class_loss + obj_loss

    def nonmax_mask(self, true_obj, true_box, pred_box):
        # YOLOv3:
        # "If the bounding box prior is not the best but does overlap a ground
        # truth object by more than some threshold we ignore the prediction.
        # We use the threshold of .5."
        # calculate the iou for each pair of pred bbox and true bbox, then find the best among them
        # we will ignore them in the object loss

        # (None, 13, 13, 3, 4)
        true_box_shape = tf.shape(true_box)
        # (None, 13, 13, 3, 4)
        pred_box_shape = tf.shape(pred_box)
        # (None, 507, 4)
        true_box = tf.reshape(true_box, [true_box_shape[0], -1, 4])
        # sort true_box to have non-zero boxes rank first
        true_box = tf.sort(true_box, axis=1, direction="DESCENDING")

        # (None, 70, 4)
        # only use maximum 70 boxes per groundtruth to calcualte IOU, otherwise
        # GPU emory comsumption would explode for a matrix like (16, 52*52*3, 52*52*3, 4)
        true_box = true_box[:, 0:70, :]
        # (None, 507, 4)
        pred_box = tf.reshape(pred_box, [pred_box_shape[0], -1, 4])

        # (None, 507, 507) for every BB prediction (13x13x3) it calculates its IoU over the ground truth
        iou = broadcast_iou(pred_box, true_box)
        # (None, 507) for every BB prediction (13x13x3) it keeps the best IoU over ground truth
        best_iou = tf.reduce_max(iou, axis=-1)
        # (None, 13, 13, 3)
        best_iou = tf.reshape(best_iou, [pred_box_shape[0], pred_box_shape[1], pred_box_shape[2], pred_box_shape[3]])
        # ignore_mask = 1 => don't ignore
        # ignore_mask = 0 => should ignore
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        # (None, 13, 13, 3, 1)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        """
        calculate loss of objectness: crossentropy
        inputs:
        true_obj: objectness from ground truth in shape of (batch, grid, grid, anchor, 1)
        pred_obj: objectness from model prediction in shape of (batch, grid, grid, anchor, 1)
        outputs:
        obj_loss: objectness loss
        """

        obj_entropy = BinaryCrossentropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

        return obj_loss + noobj_loss


    def calc_class_loss(self, true_obj, true_class, pred_class):
        """
        calculate loss of class prediction
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_class: one-hot class from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_class: one-hot class from model prediction in shape of (batch, grid, grid, anchor, num_classes)
        outputs:
        class_loss: class loss
        """
        # Yolov1:
        # "Note that the loss function only penalizes classiﬁcation error
        # if an object is present in that grid cell (hence the conditional
        # class probability discussed earlier)."
        class_loss = BinaryCrossentropy(true_class, pred_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        return class_loss

    def calc_xy_loss(self, true_obj, true_xy, pred_xy, weight):
        """
        calculate loss of the centroid coordinate: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_xy: centroid x and y from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_xy: centroid x and y from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        xy_loss: centroid loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)

        # in order to element-wise multiply the result from tf.reduce_sum
        # we need to squeeze one dimension for objectness here
        true_obj = tf.squeeze(true_obj, axis=-1)

        # YoloV1:
        # "It also only penalizes bounding box coordinate error if that
        # predictor is "responsible" for the ground truth box (i.e. has the
        # highest IOU of any predictor in that grid cell)."
        xy_loss = true_obj * xy_loss * weight

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

        return xy_loss

    def calc_wh_loss(self, true_obj, true_wh, pred_wh, weight):
        """
        calculate loss of the width and height: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_wh: width and height from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_wh: width and height from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        wh_loss: width and height loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
        return wh_loss
    
    
