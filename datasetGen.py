# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:36:08 2017

@author: sparsh
"""

"""
Crop Disease Classification Project for Code Fun Do 2017 - IIT Roorkee
"""

"""
File for transforming the dataset and preparing a csv file for the same.
"""

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Train_Dummy"
validation_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Test_Dummy"

img_width = 200
img_height = 200

#Generating the training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
print("Training data generated.\n")
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')
print("Testing data generated.\n")
print(train_generator.shape)
print("\n")
print(validation_generator.shape)