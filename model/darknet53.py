#!/usr/bin/env python
# coding: utf-8

# Utility libraries
import os
import random
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Core libraries
import numpy as np
import tensorflow as tf

from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, LeakyReLU, Add, BatchNormalization, GlobalAveragePooling2D, Dense, Softmax, Input

def Conv2D_plus(inputs, filters, kernel_size, stride = 1) -> Tensor:
    X = Conv2D(filters = filters,
                   kernel_size = kernel_size,
                   strides = stride,
                   padding = "same",
                   use_bias = False)(inputs)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    return X

def ResidualUnit(inputs, filters_alpha, filters_beta) -> Tensor:
    Y = Conv2D_plus(inputs, filters_alpha, 1)
    Y = Conv2D_plus(Y, filters_beta, 3)
    Y = Add()([Y, inputs])
    return Y

def ResidualBlock(inputs, num_filters, num_blocks) -> Tensor:
    W = Conv2D_plus(inputs, num_filters, 3, stride = 2)
    for _ in range(num_blocks):
        W = ResidualUnit(W, num_filters // 2, num_filters)
    return W

def Darknet(inputs = Input(shape=(1280, 1280, 3)),
            classification = False,
            num_classes = 10,
            trained = False) -> Model:

    Z = Conv2D_plus(inputs, 32, 3)
    Z = ResidualBlock(Z, 64, 1)
    Z = ResidualBlock(Z, 128, 2)
    Z = ResidualBlock(Z, 256, 8)
    Z = ResidualBlock(Z, 512, 8)
    Z = ResidualBlock(Z, 1024, 4)
    if classification:
        Z = GlobalAveragePooling2D()(Z)
        Z = Dense(num_classes)(Z)
        Z = Softmax()(Z)
    darknet = Model(inputs=inputs, outputs=Z, name="Darknet53")
    if trained:
        darknet = load_weights(darknet)
    return darknet


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def load_weights(model, weights_path = None, save_model = False, colab = False):
    if not weights_path:
        if colab:
            weights_path = Path('/content/drive/My Drive/ispr_yolo/weights/darknet_pretrained')
        else:
            weights_path = Path('/home/andrea/AI/ispr_yolo/weights/darknet_pretrained')
        fp = weights_path.joinpath('darknet53.weights')
    weights_array = np.fromfile(fp, dtype = np.float32, offset = 20)
    weights_num_counter = 0


    for idx in range(180):
        weights = model.get_layer(index = idx).get_weights()
        if len(weights) == 1: # It is a convolutional layer
            weights = weights[0]

            #Storing Convolutional weights
            conv_weights = weights_array[weights_num_counter + 4*weights.shape[-1] : weights_num_counter + np.prod(weights.shape) + 4*weights.shape[-1] ]
            darknet_w_shape = (weights.shape[3], weights.shape[2], weights.shape[0], weights.shape[1])
            conv_weights = np.reshape(conv_weights, darknet_w_shape)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights]

            #Storing Batch Norm weights
            batch_weights = weights_array[weights_num_counter : weights_num_counter + 4*weights.shape[-1]]
            batch_weights_array = []
            for array in np.array_split(batch_weights, 4):
                batch_weights_array.append(array)
            batch_weights_array = swapPositions(batch_weights_array, 1, 0)

            weights_num_counter += np.prod(weights.shape) + 4*weights.shape[-1]

            model.get_layer(index = idx).set_weights(conv_weights)
            model.get_layer(index = idx + 1).set_weights(batch_weights_array)

    if save_model:
        save_path = Path('/home/andrea/AI/ispr_yolo/weights')
        save_path = save_path.joinpath('darknet53.h5')
        model.save(save_path)

    return model


if __name__ == '__main__':
    model = Darknet(trained = True)
    model.summary()
