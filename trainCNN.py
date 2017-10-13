# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:23:39 2017

@author: sparsh
"""
"""
Crop Disease Classification Project for Code Fun Do 2017 - IIT Roorkee
"""
"""
File for Building the CNN Model using Keras with Theano background.
"""

import numpy as np
np.random.seed(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist

batch_size = 128
nb_classes = 10
nb_epoch = 12

#input image dimensions
img_rows = 28
img_cols = 28

#number of colvolutional filters to use
nb_filters = 32

#size of pooling area for maxpooling
pool_size = (2,2)

#convolution kernel size
kernel_size = (3,3)

"""
Load Dataset
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("Dataset Loaded.\n")

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("Dataset transformed.\n")

#Start building the model
print("Building the model.\n")
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#Define Loss function and scoring metrics
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Model built.\n")
              
#Train the model to dataset
print("Training the model.\n")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])