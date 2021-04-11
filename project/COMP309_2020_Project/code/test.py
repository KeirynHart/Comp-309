#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)
tf.random.set_seed(SEED)
print(tf.version.VERSION)

def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_path, image_size = (300, 300)):
    
    #loading test images from local directory using keras preprocessing, return test images batch.
    test_images = tf.keras.preprocessing.image_dataset_from_directory(test_path,
                                        image_size = image_size, seed = 309, labels='inferred',
                                        batch_size = 20, label_mode='categorical', shuffle = False)
    
    return test_images


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype = "float") / 255.0
    y_test = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test



def augment():
    #returning the augmentation settings so that I can plot an example of the augmentation.
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.3),
        layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.4, 0.4), width_factor=(-0.4, 0.4),
                                                           fill_mode = 'wrap')
    ]
    )
    return data_augmentation

def evaluate(X_test):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/model.h5')
    print(model.summary())
    return model.evaluate(X_test, batch_size = batch_size, verbose = 1)


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["data/test"]

    # Image size, please define according to your settings when training your model.
    image_size = (300,300)

    # Load images
    images = load_images(test_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    #X_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    #X_test = preprocess_data(X_test)
    #print(X_test.shape)
    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    loss, accuracy = evaluate(X_test)
    print("loss={}, accuracy={}".format(loss, accuracy))
