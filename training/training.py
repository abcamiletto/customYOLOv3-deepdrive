#!/usr/bin/env python

import tensorflow as tf
import sys, os, datetime
sys.path.append('/home/boscolo/ispr_yolo_light/data')
sys.path.append('/home/boscolo/ispr_yolo_light/model')

from yolo import YOLOv3
from lossfunction import YoloLoss
from DataPreprocessing import create_dataset


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2", "/gpu:3"])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    img_dir = '/home/boscolo/ispr_yolo_light/data/dataset_bdd/images/100k/train'+ '/*.jpg'
    train_ds = create_dataset(img_dir, batch = 16)

    yolo = YOLOv3(size = 1280, training = True)

    yolo_loss = [YoloLoss(10)] * 3
    nadam = tf.keras.optimizers.Nadam(learning_rate = 1e-3)
    yolo.compile(loss=[YoloLoss(10, 'dense'), YoloLoss(10, 'medium'), YoloLoss(10, 'coarse') ], optimizer=nadam)


    logdir = os.path.join("logs/fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    yolo.fit(train_ds,
             epochs = 100,
             callbacks = [tensorboard_callback])

