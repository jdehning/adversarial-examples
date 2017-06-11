import pandas as pd
import numpy as np
import keras
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from sklearn.model_selection import train_test_split
import time

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

train = pd.read_csv("data/train.csv").values
#Each line contains one image, 28x28=784 px. first Column is the number displayed.
#	print(np.shape(trainData))
#	(42000, 785)

nb_epoch = 3 # Change to 100
batch_size = 128


data_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
data_X = data_X.astype(float)
data_X /= 255.0

data_Y = keras.utils.to_categorical(train[:, 0])
nb_classes = data_Y.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size = 0.2)

def create_model_kaggle():
    nb_filters_1 = 32  # 64
    nb_filters_2 = 64  # 128
    nb_filters_3 = 128  # 256
    nb_conv = 3

    cnn = models.Sequential()

    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.Convolution2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))
    cnn.add(conv.MaxPooling2D(strides=(2,2)))

    #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    #cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu", border_mode='same'))
    #cnn.add(conv.MaxPooling2D(strides=(2,2)))

    cnn.add(core.Flatten())
    cnn.add(core.Dropout(0.2))
    cnn.add(core.Dense(128, activation="relu")) # 4096
    cnn.add(core.Dense(nb_classes, activation="softmax"))

    cnn.summary()
    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    graph = tf.get_default_graph()
    #for n in graph.as_graph_def().node:
    #    print(n.name)
    #    print(n.op)

    for lay in cnn.layers:
        print(lay.input, lay.output)

    return cnn

def create_model2():
    model = models.Sequential()
    model.add(conv.Convolution2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(conv.Convolution2D(64, (3, 3), activation='relu'))
    model.add(conv.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Dropout(0.25))
    model.add(core.Flatten())
    model.add(core.Dense(128, activation='relu'))
    model.add(core.Dropout(0.5))
    model.add(core.Dense(nb_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

model = create_model_kaggle()
time_beg = time.time()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
print(time.time() - time_beg)

model.save("keras_model1")

graph = tf.get_default_graph()


#score = model.evaluate(X_test, Y_test, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


