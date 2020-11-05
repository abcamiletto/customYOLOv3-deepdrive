
# Obstacle Detection for Autonomous Driving

A custom, from scratch, implementation of a YOLOv3-like architecture trained on Berkeley Deep Drive Dataset to detect and localize obstacle on the road.

## The Structure

You can find a full demo on this [YouTube video](https://www.youtube.com/watch?v=C2l1U2I18HQ), here's a preview:

<p align="center">
  <img width="460" height="300" src="https://github.com/abcamiletto/customYOLOv3-deepdrive/blob/master/demo.gif?raw=true">
</p>

## Main Challenges
The main problem i faced on the standard YOLOv3 (the same used on the paper) is related to the skewness of the Database. The 5 less numerous categories (Rider, Motor, Bus, Bike, Train) were never detected. 

Another challenge i had to face is the fact that the database has way more instances on the *small* size: just think about a pic of a road, the farther you look the smaller the car gets, and the more of them you'll find. So what happened is that YOLO considered equally instances of different sizes, overfitting on the small ones and undefitting the bigger.

## What i changed, and worked


When in doubt, always look at the Notebook instead of the python code: those are the one i kept on using on the final stage

## What i changed, and didn't
