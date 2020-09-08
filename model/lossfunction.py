#!/usr/bin/env python
# coding: utf-8


# Core libraries
import numpy as np
import tensorflow as tf
from utils import broadcast_iou, xywh_to_y1x1y2x2, xywh_to_x1x2y1y2

# Anchors
yolo_anchors = np.array([(10, 10), (22, 23), (47, 33), (39, 81), (82, 54), (127, 86),
                         (118, 168), (194, 130), (257, 221)], np.float32)


def decode_into_abs(y_pred, valid_anchor, num_classes = 10):
    t_xy, t_wh, objectness, classes = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1)
    # That's because it's a Logistic classifier
    objectness = tf.sigmoid(objectness)
#     classes = tf.sigmoid(classes)
# we're not gonna sigmoid 'em because we're gonna use tf.nn.softmax_cross_entropy_with_logits
    grid_size = tf.shape(y_pred)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.stack(C_xy, axis=-1)
    C_xy = tf.expand_dims(C_xy, axis=2)
##########################
    # bx = sigmoid(tx) + Cx
    # by = sigmoid(ty) + Cy
    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)
    b_xy = b_xy / tf.cast(grid_size, tf.float32)
##########################
    # it does not make sense for the box to have a negative width or height. That’s why
    # we take the exponent of the predicted number.
    b_wh = tf.exp(t_wh) * valid_anchor
##########################
    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box, objectness, classes



def encode_into_rel(y_true, valid_anchor):
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


# ## Loss Function
# Before starting with the loss function itself we need some utilities:

# In[8]:


def BinaryCrossentropy(pred_prob, labels):
    # I use a custom crossentropy because i need to weight diffently the 2 parts
    epsilon = 1e-7
    pred_prob = tf.clip_by_value(pred_prob, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(pred_prob) +
             (1 - labels) * tf.math.log(1 - pred_prob))

class YoloLoss:
    def __init__(self, num_classes, valid_anchors_wh = yolo_anchors):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5

    def __call__(self, y_true, y_pred): # In order to call it like a function
        """
        calculate the loss of model prediction for one scale
        """
        pred_box_abs, pred_obj, pred_class = decode_into_abs(
            y_pred, self.valid_anchors_wh, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)

        true_xy_abs, true_wh_abs, true_obj, true_class = tf.split(
            y_true, (2, 2, 1, self.num_classes), axis=-1)
        true_box_abs = tf.concat([true_xy_abs, true_wh_abs], axis=-1)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)

            # use the absolute yolo box to calculate iou and ignore mask
        ignore_mask = self.nonmax_mask(true_obj, true_box_abs,
                                            pred_box_abs)
######### Getting properly formatted values to process the loss functions

        true_box_rel = get_relative_yolo_box(y_true, self.valid_anchors_wh)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        # some adjustment to improve small box detection, note the (2-truth.w*truth.h) below
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        # split y_pred into xy, wh, objectness and one-hot classes
        # pred_??_rel: (batch, grid, grid, anchor, 2)
        # We have to use a sigmoid function because x,y have to be values between 0 and 1
        # cause they are realtive position INSIDE the single cell
        pred_xy_rel = tf.sigmoid(y_pred[..., 0:2])
        pred_wh_rel = y_pred[..., 2:4]

        xy_loss = self.calc_xy_loss(true_obj, true_xy_rel, pred_xy_rel, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_rel, pred_wh_rel, weight)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)


        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        return xy_loss + wh_loss + class_loss + obj_loss #, (xy_loss, wh_loss,
                                                          # class_loss,
                                                          # obj_loss)

    def nonmax_mask(self, true_obj, true_box, pred_box):
        true_box_shape = tf.shape(true_box)
        pred_box_shape = tf.shape(pred_box)
        true_box = tf.reshape(true_box, [true_box_shape[0], -1, 4])
        # sort true_box to have non-zero boxes rank first
        true_box = tf.sort(true_box, axis=1, direction="DESCENDING")

        # only use maximum 70 boxes per groundtruth to calcualte IOU, otherwise
        # GPU emory comsumption would explode for a matrix like (16, 52*52*3, 52*52*3, 4)
        true_box = true_box[:, 0:70, :]
        pred_box = tf.reshape(pred_box, [pred_box_shape[0], -1, 4])

        # (None, 507, 507) for every BB prediction (13x13x3) it calculates its IoU over the ground truth
        iou = broadcast_iou(pred_box, true_box)
        # (None, 507) for every BB prediction (13x13x3) it keeps the best IoU over ground truth
        best_iou = tf.reduce_max(iou, axis=-1)
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
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

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

        class_loss = tf.nn.softmax_cross_entropy_with_logits(true_class, pred_class, axis=-1)
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

        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
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
