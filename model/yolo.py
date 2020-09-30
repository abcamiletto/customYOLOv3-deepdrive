#!/usr/bin/env python
# coding: utf-8

# Utility libraries
from pathlib import Path

# Core libraries
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Concatenate

# Useful Darknet Functions written in Darknet53 Notebook
import darknet53
from darknet53 import Conv2D_plus, ResidualUnit, ResidualBlock



def Darknet53_yolo(pretrained = False, name = None) -> Model:
    Z = inputs = Input(shape=(None, None, 3))
    Z = Conv2D_plus(Z, 32, 3)
    Z = ResidualBlock(Z, 64, 1)
    Z = ResidualBlock(Z, 128, 2)
    Z_3 = ResidualBlock(Z, 256, 8)
    Z_2 = ResidualBlock(Z_3, 512, 8)
    Z_1 = ResidualBlock(Z_2, 1024, 4)
    darknet = Model(inputs=inputs, outputs=(Z_1,Z_2,Z_3), name=name)
    if pretrained:
        darknet = darknet53.load_weights(darknet)
    return darknet

def YoloConv2D(inputs, filters):
    A = Conv2D_plus(inputs, filters, 1)
    A = Conv2D_plus(A, filters*2, 3)
    A = Conv2D_plus(A, filters, 1)
    A = Conv2D_plus(A, filters*2, 3)
    A = Conv2D_plus(A, filters, 1)
    return A

def YoloOut(inputs, filters, anchors, classes, name = None):
    B = Conv2D_plus(inputs, filters * 2, 3)
    B = Conv2D(filters = anchors * (classes + 5),
               kernel_size = 1,
               strides = 1,
               padding = "same",
               use_bias = True)(B)
    B = tf.reshape(B, (-1,inputs.shape[1],inputs.shape[2],3,15), name = name)
    return B

def YoloUpsampling(inputs, filters):
    C = Conv2D_plus(inputs, filters, 1)
    C = UpSampling2D(2)(C)
    return C


def YOLOv3(size=None, channels=3, classes=10, training=False):
    X = inputs = Input([size, size, channels])
    Darknet = Darknet53_yolo(pretrained = True, name='darknet')
    Darknet.trainable = False
    Z_1, Z_2, Z_3 = Darknet(X)

    # Output number 1
    Z_1 = YoloConv2D(Z_1, 512)
    Z_out1 = YoloOut(Z_1, 512, 3, classes, name = 'coarser_grid')

    # Output number 2
    Z_1 = YoloUpsampling(Z_1, 256)
    Z_2 = Concatenate()([Z_1, Z_2])
    Z_2 = YoloConv2D(Z_2, 256)
    Z_out2 = YoloOut(Z_2, 256, 3, classes, name = 'medium_grid')

    # Output number 3
    Z_2 = YoloUpsampling(Z_2, 128)
    Z_3 = Concatenate()([Z_2, Z_3])
    Z_3 = YoloConv2D(Z_3, 128)
    Z_out3 = YoloOut(Z_3, 128, 3, classes, name = 'dense_grid')

    if training:
        return Model(inputs, (Z_out3, Z_out2, Z_out1), name='YOLOv3')


if __name__ == '__main__':
    yolo = YOLOv3(size = 1280,training = True)
    yolo.summary()
