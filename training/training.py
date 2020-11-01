import sys, os, datetime
sys.path.append('/home/andrea/AI/ispr_yolo/data')
sys.path.append('/home/andrea/AI/ispr_yolo/model')
from yolo import YOLOv3
from lossfunction import YoloLoss
from DataPreprocessing import create_dataset
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

tensorboard --logdir logs --port 6001 --host=0.0.0.0

def scheduler(epoch, lr):
    epoch=epoch+1
    if epoch < 4:
        print('Learning rate for epoch n.', epoch, ' :', lr)
        return lr
    elif epoch < 8:
        print('Learning rate for epoch n.', epoch, ' :', lr*0.8)
        return lr * 0.8
    else:
        print('Learning rate for epoch n.', epoch, ' :', lr*0.2)
        return lr * 0.2

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

with tf.device('/cpu:0'):
    img_dir = '/home/andrea/AI/ispr_yolo/data/dataset_bdd/images/100k/train' + '/*.jpg'
    train_ds_aug = create_dataset(img_dir, batch = 24, augmented = True)
    img_dir = '/home/andrea/AI/ispr_yolo/data/dataset_bdd/images/100k/val' + '/*.jpg'
    val_ds = create_dataset(img_dir, batch = 24, validation = True)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2", "/gpu:3"])
with strategy.scope():
    log_dir = os.path.join("logs/fit", datetime.datetime.now().strftime("%m%d-%H%M%S"))
    yolo = YOLOv3(size = 1280, training = True)
    nadam = tf.keras.optimizers.Nadam(learning_rate=1e-03) #2e-04
    yolo.compile(loss=[YoloLoss(10, 'dense', softmax = False),
                       YoloLoss(10, 'medium', softmax = False),
                       YoloLoss(10, 'coarse', softmax = False)],
                       optimizer=nadam)


    log_dir = os.path.join("logs/fit", datetime.datetime.now().strftime("%m%d-%H%M%S"))
    logging = TensorBoard(log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

    yolo.fit(train_ds_aug,
             epochs = 20,
             callbacks = [logging, checkpoint, lr_scheduler, early_stopping],
             validation_data = val_ds)
