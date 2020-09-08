#!/usr/bin/env python
# coding: utf-8



# Importing main packages
import numpy as np
import tensorflow as tf



BATCH_SIZE = 32
IMG_DIMENSION = 1280
SCALE = 1
SMALL = 35
MEDIUM = 70
LARGE = 140

yolo_anchors = tf.constant([(10, 10), (22, 23), (47, 33), (39, 81), (82, 54), (127, 86),
                         (118, 168), (194, 130), (257, 221)], tf.float32) / IMG_DIMENSION
masks = [[0,1,2],
         [3,4,5],
         [6,7,8]]



class Preprocess:
    def __init__(self, num_classes = 10, output_dimension = 1280, training = True):
        self.num_classes = num_classes
        self.output_dimension = output_dimension
        self.training = training

    def __call__(self, image_path):
        '''
        INPUT: the path of the image i want to preprocess
        OUTPUTS: a Tensor if training is false
                 a Tuple of (1920x1920, (13x13x3x15, 26x26x2x15, 52x52x3x15))
                            (Image, Ground Truth)
        '''
        img = self.load_img(image_path)
        label = self.load_label(image_path)
        img, label = self.resize_img_n_label(img, label)

        output = (img, (self.preprocess_label_for_one_scale('large', label),
              self.preprocess_label_for_one_scale('medium', label),
              self.preprocess_label_for_one_scale('small', label)))

        return output


    def preprocess_label_for_one_scale(self, grid_size, label):
        if grid_size == 'small':
            grid_size = SMALL
            idx = 0
        elif grid_size == 'medium':
            grid_size = MEDIUM
            idx = 1
        elif grid_size == 'large':
            grid_size = LARGE
            idx = 2
        else: raise ValueError('expected small, medium or large')


        cell = tf.cast(label[..., :2]*grid_size,tf.int64)
        anc = self.find_best_anchor(label)
        mask = tf.equal(anc // 3, idx)
        anc = anc // 3
        cell = tf.concat([cell, tf.expand_dims(anc, 1)], 1)
        cell_filtered = tf.boolean_mask(cell, mask, axis = 0)
        label_filtered = tf.boolean_mask(label, mask, axis = 0)
        pro_label = tf.scatter_nd(cell_filtered, label_filtered, [grid_size, grid_size, 3, 15])

        return pro_label

    def resize_img_n_label(self, img, label):
        img = tf.image.resize_with_pad(img, 1280, 1280)
        y_shift = (1280-720)/2

        new_label = tf.stack([label[..., 0], label[..., 1]+y_shift], 1)
        new_label = tf.concat([new_label, label[..., 2:]], 1)
        label = tf.concat([new_label[..., 0:4]/IMG_DIMENSION, new_label[..., 4:]], axis = 1)

        return img, label

    def find_best_anchor(self, bb, anchors = yolo_anchors):
        ## assuming bb is like [x,y,w,h]
        bb = bb[..., 2:4]
        bb = tf.expand_dims(bb, 1)
        ## we can just assume all anchors and the ground truth box share the same centroid. And with this assumption
        ## the degree of matching would be the overlapping area, which can be calculated by min width * min height.
        intersection = tf.math.minimum(anchors, bb)
        intersection = intersection[..., 0] * intersection[..., 1]
        union = anchors[..., 0] * anchors[..., 1] + bb[..., 0] * bb[..., 1] - intersection
        broadcast_IoU = intersection/union
        best_one = tf.math.argmax(broadcast_IoU, 1)

        return best_one

    def load_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3) #(720, 1280, 3)
        return img

    def load_label(self, img_path):
        # Get the right path
        label_path = self.get_label_path(img_path)
        # Load the file and shape it
        label = tf.io.read_file(label_path)
        label = tf.io.decode_raw(label, tf.int16)
        label = tf.cast(label,tf.float32)
        label = tf.reshape(label, [-1, 15])
        return label

    def get_label_path(self, img_path):
        parts = tf.strings.split(img_path, sep = '/images/100k/train/')
        label_path = tf.strings.join([parts[0], '/labels/train_label_raw/', parts[1], '.rawlabel'])
        return label_path



def create_dataset(imgdir):
    dataset = tf.data.Dataset.list_files([imgdir])
    mapping_func = Preprocess()
    dataset = dataset.map(mapping_func)
    dataset = dataset.batch(32).prefetch(1)
    return dataset
