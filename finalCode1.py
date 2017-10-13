# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:49:10 2017

@author: sparsh
"""

"""
Crop Disease Classification Project for Code Fun Do 2017 - IIT Roorkee
"""

"""
File for transforming the dataset and training the model.
"""
import os
os.environ['THEANO_FLAGS'] = "device=gpu1, floatX=float32"
import theano
import numpy as np
np.random.seed(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th')

#Training and Testing data locations
train_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Train_Dummy"
test_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Test_Dummy"

#input image dimensions
img_width = 200
img_height = 200
input_shape = (3, img_height, img_width)

#Model parameters
batch_size = 128
nb_classes = 4
nb_epoch = 1
samples_per_epoch = 40
nb_val_samples = 40

#size of pooling area for maxpooling
pool_size = (2,2)

#convolution kernel size
kernel_size = (3,3)
print("All Parameters initialized... Building the Model now...\n")

#Building the model
model = Sequential()
model.add(Convolution2D(32, kernel_size[0], kernel_size[1], 
                        border_mode = 'valid', 
                        input_shape = input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Flatten())
model.add(Dense(output_dim = 64))
model.add(Activation('relu'))
model.add(Dense(output_dim = 64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adadelta', 
              metrics = ['accuracy'])
print("Model built successfully... Generating dataset now...\n")

#Initialize Training and Testing Dataset Generator
train_datagen = ImageDataGenerator(
        rescale = 1./255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
print("Generators Initialized...\n")

#Generate Training and Testing Data
train_generator = train_datagen.flow_from_directory(
        train_data_dir, 
        target_size = (img_height, img_width), 
        batch_size = batch_size, 
        class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(
        test_data_dir, 
        target_size = (img_height, img_width), 
        batch_size = batch_size, 
        class_mode = 'categorical')
print("Dataset generated... Fitting the model now...\n")

#Fit the model to training data
model.fit_generator(
        train_generator, 
        samples_per_epoch = samples_per_epoch, 
        nb_epoch = nb_epoch, 
        validation_data = test_generator, 
        nb_val_samples = nb_val_samples)
print("Model fit successful... Saving the model weights...\n")

#Save the weights
model.save_weights('first_try.h5')
print("Program Execution Complete...\n")