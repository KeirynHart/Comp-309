#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)
tf.random.set_seed(SEED)


def load_train_images(train_path, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    #splitting the data into training and validation sets.
    #batch size which is the number of training examples in one iteration of the model.
    train_batch = tf.keras.preprocessing.image_dataset_from_directory(train_path, subset='training', validation_split = 0.3,
                                          image_size = image_size, seed = 309, labels='inferred',
                                          batch_size = 20, label_mode = 'categorical', shuffle = True)

    val_batch = tf.keras.preprocessing.image_dataset_from_directory(train_path, subset='validation', validation_split = 0.3,
                                        image_size = image_size, seed = 309, labels='inferred',
                                        batch_size = 20, label_mode='categorical', shuffle = True)

    return train_batch, val_batch


def preprocess_data(X):
    
    #data_augmentation step, flipping the images horizontally, giving the images random roations and random translations meaning
    #moving the images slightly in different directions.
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.3),
        layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.4, 0.4), width_factor=(-0.4, 0.4),
                                                           fill_mode = 'wrap')
    ]
    )
    #using the above augmentation parameters to apply it to the training data.
    augmented_train_data = X.map(
      lambda x, y: (data_augmentation(x, training=True), y))
    
    return X

def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    
    model = Sequential()
    
    #first convolutionary layer and pooling layer with filter size 32:
    #relu activation function which is considered a default for CNN's like this, has the ability to output true 0 which
    #simplifies the model and increases the speed.
    #padding means that the output layer will be the same size as the input layer, essentially keeps the output size the same
    #as input.
    #kernal size refers the the dimension of the filter's mask.
    #filter size is the number of filters in the convolution layer, generally want to increase through the model.
    model.add(Conv2D(32, kernel_size = (3,3), padding = 'same', input_shape = (300,300,3)))
    model.add(Conv2D(32, kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #second convolutionary layer and pooling layer with increased filter size of 64:
    model.add(Conv2D(64, kernel_size = (3,3)))
    model.add(Conv2D(64, kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #third convolutionary and pooling layer, increased filter size to 128:
    model.add(Conv2D(128, kernel_size = (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #flatten the model so it removes all of the dimensions except for one.
    model.add(Flatten())
    
    #adding a dropout for training, three dense layers along with softmax activation:
    model.add(Dense(units = 64))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.1))
    
    model.add(Dense(units = 128))
    model.add(Dense(units = 3))
    model.add(Activation('softmax'))
    
    #adam optimizer, good for noisy data, adaptive learning rate, generally a good optimizer.
    #categorical cross entropy, good for categorical data i.e. 3 classes, gives differences between class probabilities and
    #returns the class with the highest probability.
    #metrics - accuracy, what we are measuring
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate = 0.0001),
              metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, train, val):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    #creating a model checkpoint to save the model only if it is better than the previous (dont think this works)
    file_path = '/Users/keirynhart/Documents/Uni/Comp 309/project/COMP309_2020_Project/Checkpoint'
    check = ModelCheckpoint(filepath = file_path, save_freq = 5,
                       save_best_only = True, verbose = 1)
    
    #fitting the model with augmented data, validation data and 15 epochs along with the callback defined above.
    #epochs being the number of times the training data is passed over by the model.
    model_fit = model.fit(train, validation_data = val,
                          epochs = 15, callbacks = [check], verbose = 1)
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    #training path
    train_path = "data/Train_data"
    
    #loading the images
    train_batch, val_batch = load_train_images(train_path)
    
    #augmentation
    train_batch = preprocess_data(train_batch)
    
    #construct model
    model = construct_model()
    
    #train the model
    model = train_model(model, train_batch, val_batch)
    
    #save
    save_model(model)
