#!/usr/bin/env python
# coding: utf-8

# # K-Means on dataset Bounding Boxes
# Since we need to get 3 anchors per output we decided to follow the original paper approach of studying the training dataset to get meaningful results

# In[4]:


#Importing needed packages
import numpy as np
import pathlib
from sklearn.cluster import KMeans
from DatasetFormatting import json_parser, json_cleaner


# ## Loading BB
# We're gonna load all bounding boxes in a NumPy array and check its shape

# In[5]:


main_folder = str(pathlib.Path.cwd().joinpath('dataset_bdd', 'labels'))
json_train_dir = main_folder + '/bdd100k_labels_images_train.json'
data_train = json_parser(json_train_dir)
data_train = json_cleaner(data_train)


# In[6]:


pos_counter = 0
bb_array =  []
for item in data_train:
    for obj in item['labels']:
        bb_array.append([obj['box2d']['wb'],obj['box2d']['hb']])
        pos_counter += 1

bounding_boxes = np.array(bb_array)
bounding_boxes.shape


# ## Performing the calculation
# May take a while!

# In[11]:


kmeans = KMeans(n_clusters = 9)
y_pred = kmeans.fit_predict(bounding_boxes)


# In[12]:


anchors = np.around(kmeans.cluster_centers_).tolist()
anchors = sorted(anchors, key = lambda x: x[0]*x[1])
anchors = [(round(x[0]),round(x[1])) for x in anchors]
print(anchors)


# In[14]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

some_ones = bounding_boxes[::200]
pred = y_pred[::200]
colors = ['r','b','g','purple','y','orange', 'b', 'c', 'crimson']

plt.style.use('ggplot')
for idx, bb in enumerate(some_ones):
    plt.plot(bb[0], bb[1], color = colors[pred[idx]], marker = '.')
plt.axis([-5,200,-5,200])
plt.xlabel('Width')
plt.ylabel('Height')

