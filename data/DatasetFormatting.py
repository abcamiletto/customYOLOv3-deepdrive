#!/usr/bin/env python
# coding: utf-8

# # Dataset Formatting
# The Berkeley Deep Drive dataset is surely cool, but we need to filter the enormous amount of info that are given with each image.

# In[2]:


#Importing main packages

import numpy as np
import tensorflow as tf
import cv2
import pathlib
import json


def pov_change(x1,x2,y1,y2):
    xb = (x1+x2)/(2)
    yb = (y1+y2)/(2)
    wb = abs(x1-x2)/2
    hb = abs(y1-y2)/2
    return xb,yb,wb,hb

#Loading a JSON file into a Python variable
def json_parser(path):
    with open(path, 'r') as read_file:
        data = json.load(read_file)
    return data


# In[5]:


#Cleaning a JSON file from all the useless information
def json_cleaner(data):
    # Cleaning timestamps and not so useful attributes
    for item in data:
        del item['attributes']
        del item['timestamp']

    # Cleaning drivable area and lanes
    for item in data:
        storing_indexes = []
        for index, i in enumerate(item['labels']):
            del i['attributes']
            del i['manualShape']
            del i['manualAttributes']
            if 'box2d' in i:
                xb,yb,wb,hb = pov_change(i['box2d']['x1'],i['box2d']['x2'],i['box2d']['y1'],i['box2d']['y2'])
                i['box2d']['xb'] = round(xb,2)
                i['box2d']['yb'] = round(yb,2)
                i['box2d']['wb'] = round(wb,2)
                i['box2d']['hb'] = round(hb,2)
                del i['box2d']['x1']
                del i['box2d']['x2']
                del i['box2d']['y1']
                del i['box2d']['y2']
            # Checking if anything is corrupted
            if not 'poly2d' in i and not 'box2d' in i: print('no box2d?' + str(i['id']))
            if 'box3d' in i: print('wtf')

            del i['id']
            if i['category'] == 'lane' or i['category'] == 'drivable area':
                storing_indexes.append(index)
        storing_indexes.sort(reverse=True)
        for indexes in storing_indexes:
            del item['labels'][indexes]
    return data


def split_data(data,path):
    for item in data:
        name = item['name']
        with open(path + name + '.json', 'w') as file_to_write:
            json.dump(item, file_to_write, indent = 4)
    return

runtime = False
if runtime:
    data_train = json_parser(json_train_dir)
    data_train = json_cleaner(data_train)
    split_data(data_train, folder_train)
    print('Training data: done')
    data_val = json_parser(json_val_dir)
    data_val = json_cleaner(data_val)
    split_data(data_val, folder_val)
    print('Validation data: done')

def detect_missing_labels(delete = False):
    counter = 1
    main_dir = pathlib.Path.cwd().joinpath('dataset_bdd')
    img_dir = main_dir.joinpath('images', '100k', 'train').glob('*.jpg')
    label_dir = main_dir.joinpath('labels', 'train_jsons')
    for img in img_dir:
        img_name = img.name
        label_path = label_dir.joinpath(img_name + '.json')
        if not label_path.is_file():
            counter = counter + 1
            if delete:
                img.unlink()
                print('deleted')
    print('Images w/o label:' + str(counter))

def showme_data_format(path):
    with open(path, 'r') as read_file:
        data = json.load(read_file)
    print('Data type: ' + str(type(data)))
    print('Element of the list: ' + str(type(data[0])))
    print('Keys of the dictionaries: ')
    for key in data[0]:
        print('    ' + str(key))
    print('Dict example:')
    for key, value in data[0].items():
        if key != 'labels':
            print('    Key: ' + str(key))
            print('      Value: ' + str(value))
        else:
            print('    Key: ' + str(key))
            print('      Value: it is a ' + str(type(value)) + ' made of ' + str(type(value[0])))
            for ondex, obj in enumerate(value):
                if ondex < 2: print(obj)
    print('\n\n\n')
