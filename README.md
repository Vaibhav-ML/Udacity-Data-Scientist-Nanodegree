# Udacity-Data-Scientist-Nanodegree
Object Detection in an Urban Environment
Goal
To classify and localize the cars, pedestrians and cyclists in camera input feed.

Data
For this project, we will be using data from the Waymo Open dataset.

Structure
Data
The data we will use for training, validation and testing is organized as follow:

/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
The training_and_validation folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The testing folder contains frames from the 10 fps video without downsampling.
