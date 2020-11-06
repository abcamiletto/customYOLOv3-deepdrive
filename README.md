# Obstacle Detection for Autonomous Driving

A custom, from scratch, implementation of a YOLOv3-like architecture trained on Berkeley Deep Drive Dataset to detect and localize obstacle on the road.

## The Structure
Since the original papers are far from clear and it's not that easy to understand what's the yolo architecture i decided to make my own scheme.

<p align="center">
  <img width="1280" src="https://github.com/abcamiletto/customYOLOv3-deepdrive/blob/master/yolo_black.png?raw=true">
</p>
Overall we have ~60M parameters, 40M of them are used in Darknet53 which is the backbone structure pretrained on the Image-Net data.

## Results
You can find a full demo on this [YouTube video](https://www.youtube.com/watch?v=C2l1U2I18HQ), here's a preview:

<p align="center">
  <img width="460" height="300" src="https://github.com/abcamiletto/customYOLOv3-deepdrive/blob/master/demo.gif?raw=true">
</p>

I achieved a discrete result of .55 F1 accuracy over 5000 images, 0.7 if we consider only object with a bounding box of over 2000 pixel squared

## Main Challenges
The main problem i faced on the standard YOLOv3 (the same used on the paper) is related to the skewness of the Database. The 5 less numerous categories (Rider, Motor, Bus, Bike, Train) were never detected. 

Another challenge i had to face is the fact that the database has way more instances on the *small* size: just think about a pic of a road, the farther you look the smaller the car gets, and the more of them you'll find. So what happened is that YOLO considered equally instances of different sizes, overfitting on the small ones and undefitting the bigger.

## What i changed, and worked
The main improvements came with:

 - **Softmax Layer**: even if i ended up with somewhat worse overall performance, its addition helped a lot in the detection of the less numerous classes, which before weren't detected at all
 - **Re-Balancing outputs**: i over-weighted the loss function on the coarser grid, which is responsible of the detection of bigger objects and underweighted the denser one.
 - **Delayed Esponential Decay of the learning rate**: even if it's not the one used in the papers, it worked better for me.

## What i changed, and didn't

 - **Adding a convolutional layer**: it might seems a wise choice since the images we're using are way bigger that the ones used by the original paper. It turns out it isn't since the bounding boxes are *more or less* the same, as it can be seen by a K-means analysis
 - **1-cycle learning rate scheduler**: or sort of. It's what has been proposed on the original papers and it did not worked at all for me.

## Overall take on the code
When in doubt, always look at the Notebook instead of the python code: those are the one i kept on using on the final stage!

### References
There are mainly 2 repo i looked into and i want to thank:

 - https://github.com/ethanyanjiali/deep-vision
 
 - https://github.com/qqwweee/keras-yolo3
