# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 02:39:55 2017

@author: sparsh
"""

"""
Crop Disease Classification Project for Code Fun Do 2017 - IIT Roorkee
"""

"""
File for predicting a test image.
"""

import os
os.environ['THEANO_FLAGS'] = "device=gpu1, floatX=float32"
import theano
import numpy as np
np.random.seed(1)

import pandas as pd
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from PIL import Image
K.set_image_dim_ordering('th')

#Path to model weights file
weights_path = "E:\\Interesting\\Code Fun Do 2017\\vgg16_weights.h5"
top_model_weights_path = "E:\\Interesting\\Code Fun Do 2017\\bottleneck_fc_model.h5"

#Unknown Image Location
validation_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Trial\\cercospora_leaf_spot_365.jpg"
#validation_data_dir = "E:\\Interesting\\Code Fun Do 2017\\Trial"

#input image dimensions
img_width = 200
img_height = 200
input_shape = (3, img_height, img_width)

#Model parameters
batch_size = 32
nb_classes = 4
nb_epoch = 3
nb_train_samples = 50
nb_validation_samples = 25

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), "Model weights not found (see 'weights_path' variable in script)."
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print("Model loaded.\n")

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)
#print("DC.\n")
print("Final Model Assembled.\n")

#datagen = ImageDataGenerator(rescale=1./255)
#generator = datagen.flow_from_directory(
#            validation_data_dir,
#            target_size=(img_width, img_height),
#            batch_size=32,
#            class_mode=None,
#            shuffle=False)
#bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
#np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
#print("Testing features stored.\n")

#data = np.load(open('bottleneck_features_validation.npy'))
img = Image.open(validation_data_dir)

img.load()
#print("chutiya.\n")
data = np.asarray(img, dtype="int32")
#print("harami.\n")
print(data.shape)
data = data.reshape(1, 3, 200, 200)
print("Prediction begins.\n")
output = model.predict_classes(data, batch_size=32, verbose=1)
print(output)