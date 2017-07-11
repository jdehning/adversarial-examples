####################################################
# This program creates a convolutional network to
# classify the MNIST dataset. The code is originally
# based on a code from kaggle.com
####################################################


import pandas as pd
import numpy as np
import keras
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from sklearn.model_selection import train_test_split
import time

# import this functions to use them later
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

####################################################
# load trainingsdata as ? pands dataframe ?
####################################################
train = pd.read_csv("data/train.csv").values
#Each line contains one image, 28x28=784 px. first Column is the number displayed.
#	print(np.shape(trainData))
#	(42000, 785)

nb_epoch = 20
batch_size = 128

####################################################
# manipulate data to fit our needs
####################################################
data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)    #just the picture, since first column is label
data_X = data_X.astype(float)                           
data_X /= 255.0                                             #normalise greyscale values to 1                                         

####################################################
# create labels
####################################################
data_Y = keras.utils.to_categorical(train[:, 0])
nb_classes = data_Y.shape[1]                                #total number of classes

####################################################
# split data in trainings and test data
####################################################
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size = 0.2)


####################################################
# create model for neural network
####################################################
def create_model_1():
    nb_filters_1 = 32   # number of filters for first convolutional layer
    nb_filters_2 = 64   # number of filters for second convolutional layer
    nb_conv = 3         #   

    cnn = models.Sequential()

    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(core.Flatten())
    cnn.add(core.Dropout(0.2))
    cnn.add(core.Dense(128, activation="relu")) # 4096
    cnn.add(core.Dense(nb_classes, activation="softmax"))

    cnn.summary()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return cnn

####################################################
# create model for neural network
####################################################
def create_model_2():
    nb_filters_1 = 32   # number of filters for first convolutional layer
    nb_filters_2 = 64   # number of filters for second convolutional layer
    nb_conv = 3         #   

    cnn = models.Sequential()

    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(core.Flatten())
    cnn.add(core.Dropout(0.2))
    cnn.add(core.Dense(128, activation="relu")) # 4096
    cnn.add(core.Dense(nb_classes, activation="softmax"))

    cnn.summary()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return cnn

####################################################
# create model for neural network
####################################################
def create_model_3():
    nb_filters_1 = 32   # number of filters for first convolutional layer
    nb_filters_2 = 64   # number of filters for second convolutional layer
    nb_conv = 3         #   

    cnn = models.Sequential()

    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(core.Flatten())
    cnn.add(core.Dropout(0.2))
    cnn.add(core.Dense(128, activation="relu")) # 4096
    cnn.add(core.Dense(nb_classes, activation="softmax"))

    cnn.summary()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return cnn

####################################################
# let it run
####################################################
#model = create_model_1()
model = keras.models.load_model('mnist_models/mnist_model2')
####################################################
# callbacks are functions that are called after
# every epoch
####################################################
callbacks = [
        # early stopping checks wether the loss function did decrease in the
        # last two epochs. if not the training is stoppped
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        # creates an output of the model every 10 epochs and only saves the
        # best model according to loss function
        ModelCheckpoint('mnist_model2b_check', monitor='val_loss', save_best_only=True, verbose=1, period=1),
        ]
    
#callbacks = [EarlyStopping(monitor='val_loss',patience=2,verbose=0)]

####################################################
# start training
####################################################
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=callbacks, validation_data=(X_test, Y_test))

####################################################
# dump final model
####################################################
model.save("mnist_model2b")

print(hist.history)



score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


