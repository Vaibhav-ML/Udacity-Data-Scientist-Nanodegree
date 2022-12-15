# Object Detection in an Urban Environment
# Goal
To classify and localize the cars, pedestrians and cyclists in camera input feed.

# Data

For this project, we will be using data from the Waymo Open dataset.

# Config

Use the config in Experiment_2 folder for best results.

# Structure

The data we will use for training, validation and testing is organized as follow:

/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
    
The training_and_validation folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The testing folder contains frames from the 10 fps video without downsampling.

# Preprocessing

Impment the create_splits.py file and then run :

python create_splits.py --data-dir /home/workspace/data

This will create the train, test and val split.
### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:

python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt

A new config file has been created, `pipeline_new.config`.

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder.

# Model Training and Evaluation

Launch the training process:

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

python experiments/model_main_tf2.py --model_dir=experiments/Experiment_1/ --pipeline_config_path=experiments/Experiment_1/pipeline_new.config

To monitor the training, you can launch a tensorboard instance by running python -m tensorboard.main --logdir experiments/reference/.

# Evaluation

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/

# Creating Animation

Export the trained model
Modify the arguments of the following function to adjust it to your models:

python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/

This should create a new folder experiments/reference/exported/saved_model. You can read more about the Tensorflow SavedModel format here.

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):

python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif



