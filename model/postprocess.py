# Core libraries
import numpy as np
import tensorflow as tf
import sys, os, datetime
sys.path.append('/home/andrea/AI/ispr_yolo/data')
sys.path.append('/home/andrea/AI/ispr_yolo/model')
from DataPreprocessing import generate_one_example, Preprocess
from utils import broadcast_iou, xywh_to_y1x1y2x2, xywh_to_x1x2y1y2
from lossfunction import decode_output, from_rel_to_abs
import matplotlib.pyplot as plt
import cv2 as cv

# Anchors
yolo_anchors = tf.constant([(19, 19), (43, 46), (94, 66), (77, 163), (163, 107), (253, 172), (237, 337), (388, 260), (514, 441)], tf.float32)

class PostProcessor():
    def __init__(self, iou_thresh, score_thresh, ground = False, max_detection=100, num_classes = 10):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_detection = max_detection
        self.num_classes = num_classes
        self.ground = ground
        self.class_names = ['bus', 'light', 'sign', 'person', 'bike',
                            'truck', 'motor', 'car', 'train', 'rider']

    def __call__(self, raw_yolo_outputs):
        boxes, objectness, class_probs = self.preprocess(raw_yolo_outputs)
        objectness, classes = self.split_multilabel(objectness, class_probs)
        
        final_boxes = self.non_max_sup(boxes, objectness, classes)

        return final_boxes
    
    def split_multilabel(self, objectness, classes):
        classes = tf.argmax(classes, axis = 1)
        full_classes = np.tile(classes, [10,1])
        full_objectness = np.tile(objectness, [10,1])
        for i in range(10):
            mask = np.equal(full_classes[i], i)
            full_classes[i] = full_classes[i]*mask
            full_objectness[i] = full_objectness[i]*mask
        return full_objectness, full_classes
            
    
    def non_max_sup(self, boxes, scores, class_probs):
        output = {}
        for i in range(10):
            selected_indices = tf.image.non_max_suppression(boxes, scores[i], 30, iou_threshold=self.iou_thresh, score_threshold=self.score_thresh)
            selected_boxes = tf.gather(boxes, selected_indices)
            selected_scores = tf.gather(scores[i], selected_indices)
            output[self.class_names[i]] = (selected_boxes, selected_scores)
        
        return output
    

        
    def preprocess(self, raw_yolo_outputs):
        boxes, objectness, class_probs = [], [], []
        
        for idx, out in enumerate(raw_yolo_outputs):
            if not self.ground: out = decode_output(out, yolo_anchors[3*idx:3*idx+3])
            y_box_abs, y_obj, y_class = from_rel_to_abs(out)
            y_box_abs = xywh_to_x1x2y1y2(y_box_abs)
            
            boxes.append(tf.reshape(y_box_abs, (-1, 4)))
            objectness.append(tf.reshape(y_obj, (-1)))
            class_probs.append(tf.reshape(y_class, (-1,10)))
            
        boxes = tf.concat(boxes, axis=0)
        objectness = tf.concat(objectness, axis=0)
        class_probs = tf.concat(class_probs, axis=0)
        return boxes, objectness, class_probs
    
def draw_boxes(image, boxes):
    color={
        'bus' : (1, 1, 0.4),
        'light' : (1, 0.2, 0.2),
        'sign' : (0.8, 0.8, 0.8),
        'person' : (0, 1, 0.5),
        'bike' : (0.8, 1, 0.6),
        'truck': (1, 0.6, 0.8),
        'motor' : (0.8, 0.8, 1),
        'car' : (0.2, 0.2, 1),
        'train' : (0.2, 1, 0),
        'rider' : (0.4, 0.4, 0.4)
    }
    image = np.float32(image)
    for classes, box_n_score in boxes.items():
        boxes, scores = box_n_score
        if boxes.shape[0] > 0:
            for box, score in zip(boxes,scores):
                pt1 = tuple(np.int32(box[0:2]*1280))
                pt2 = tuple(np.int32(box[2:4]*1280))
                image = cv.rectangle(image, pt1, pt2, color[classes], thickness=2)
                area = (pt1[0]-pt2[0])*(pt1[1]-pt2[1])
                if area > 1600:
                    pt1 = (pt1[0], pt1[1]-3)
                    text = classes + ': ' + str(round(float(score), 2))
                    image = cv.putText(image, text, pt1 , cv.FONT_HERSHEY_SIMPLEX, 0.7, (1,1,1), 2)
        else:
            pass
    return image

def show_me(model, path):
    img, label_ground = prep(image_path)
    
    img = tf.expand_dims(img, 0)
    label = yolo.predict(img)
    label = [tf.squeeze(x, axis = 0) for x in label]
    
    ground_output = post(label_ground)
    model_output = post_model(label)
    img = tf.squeeze(img, 0)
    return img, model_output, ground_output

def create_model(weights):
    yolo = YOLOv3(size = 1280, training = True)
    yolo.load_weights(weights)
    return yolo

from yolo import YOLOv3