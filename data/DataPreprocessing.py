#!/usr/bin/env python
# coding: utf-8



# Importing main packages
import numpy as np
import tensorflow as tf



BATCH_SIZE = 32
IMG_DIMENSION = 1280
SCALE = 1
COARSE = 40
MEDIUM = 80
DENSE = 160

yolo_anchors = tf.constant([(19, 19), (43, 46), (94, 66), (77, 163), (163, 107), (253, 172), (237, 337), (388, 260), (514, 441)], tf.float32) / IMG_DIMENSION
masks = [[0,1,2],
         [3,4,5],
         [6,7,8]]



class Preprocess:
    def __init__(self, num_classes = 10, output_dimension = 1280, training = True, validation = False, augmentation = False, test = False):
        self.num_classes = num_classes
        self.output_dimension = output_dimension
        self.training = training
        self.validation = validation
        self.test = test
        self.augmentation = augmentation

    def __call__(self, image_path):
        '''
        INPUT: the path of the image i want to preprocess
        OUTPUTS: a Tensor if training is false
                 a Tuple of (1280x1280, (35x35x3x15, 70x70x3x15, 140x140x3x15))
                            (Image, Ground Truth)
        '''
        img = self.load_img(image_path)
        label = self.load_label(image_path)
        
        if self.augmentation:
            coin = tf.random.uniform([1])
            if coin > 0.5:
                img, label = self.augmentation_func(img, label)
            img = tf.image.random_brightness(img, max_delta=25.0 / 255.0)
            img = tf.image.random_saturation(img, lower=0.75, upper=1.25)
        
        img, label = self.resize_img_n_label(img, label)
        
        output = (img, (self.preprocess_label_for_one_scale('dense', label),
              self.preprocess_label_for_one_scale('medium', label),
              self.preprocess_label_for_one_scale('coarse', label)))

        return output

    def augmentation_func(self, img, label):
        flipped_img = tf.image.flip_left_right(img)
        bb_n = tf.shape(label)[0]
        offset = tf.fill([bb_n, 1], 1280)
        offset = tf.cast(offset,tf.float32)
        x = tf.expand_dims(label[...,0], axis = 1)
        flipped_label = tf.concat([offset-x, label[..., 1:]], axis = -1)
        return flipped_img, flipped_label

    def preprocess_label_for_one_scale(self, grid_size, label):        
        if grid_size == 'dense':
            grid_size = DENSE
            idx = 0
        elif grid_size == 'medium':
            grid_size = MEDIUM
            idx = 1
        elif grid_size == 'coarse':
            grid_size = COARSE
            idx = 2
        else: raise ValueError('expected small, medium or large')

        #those are the indices in which i'll place the label
        cell = tf.cast(label[..., :2]*grid_size,tf.int64)
        cell_x = cell[..., 0]
        cell_y = cell[..., 1]
        cell = tf.stack([cell_y, cell_x], axis = -1)
        anc = self.find_best_anchor(label)
        mask = tf.equal(anc // 3, idx)
        anc = anc % 3
        cell = tf.concat([cell, tf.expand_dims(anc, 1)], 1)
        cell_filtered = tf.boolean_mask(cell, mask, axis = 0)
        label_filtered = tf.boolean_mask(label, mask, axis = 0)
        ### getting x,y relative values to cell borders
        label_filtered = tf.concat([label_filtered[..., :2]*grid_size - tf.floor(label_filtered[..., :2]*grid_size), label_filtered[..., 2:]], 1)
        pro_label = tf.scatter_nd(cell_filtered, label_filtered, [grid_size, grid_size, 3, 15])

        return pro_label

    def resize_img_n_label(self, img, label):
        img = tf.image.resize_with_pad(img, 1280, 1280)
        img = img / 255

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

############## Loading images and labels

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
        if not self.validation and not self.test:
            parts = tf.strings.split(img_path, sep = '/images/100k/train/')
            label_path = tf.strings.join([parts[0], '/labels/train_label_raw/', parts[1], '.rawlabel'])
        elif self.validation:
            parts = tf.strings.split(img_path, sep = '/images/100k/val/')
            label_path = tf.strings.join([parts[0], '/labels/val_label_raw/', parts[1], '.rawlabel'])
        elif self.test:
            parts = tf.strings.split(img_path, sep = '/images/100k/test/')
            label_path = tf.strings.join([parts[0], '/labels/val_label_raw/', parts[1], '.rawlabel'])
        return label_path
    



def create_dataset(global_path, batch = 32, validation = False, example = False, augmented = False):   
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = tf.data.Dataset.list_files([global_path])
    mapping_func = Preprocess(validation = validation)
    if augmented:
        mapping_func_augm = Preprocess(augmentation = True)
        train_ds = data.map(mapping_func_augm)
    else: train_ds = data.map(mapping_func)
    if example:
        return train_ds
    train_ds = train_ds.batch(batch).prefetch(1)
    return train_ds

def generate_two_label_example(batch = 1):
    dataset = create_dataset('/home/andrea/AI/ispr_yolo/data/dataset_bdd/images/100k' + '/train/*.jpg', batch = batch, example = True)
    labels = []
    for item in dataset.take(2):
        img, label = item
        labels.append(label[2])
    return labels

def generate_one_example(batch = 1):
    dataset = create_dataset('/home/andrea/AI/ispr_yolo/data/dataset_bdd/images/100k' + '/train/*.jpg', batch = batch, example = True)
    labels = []
    for item in dataset.take(1):
        img, label = item
        return img, label

