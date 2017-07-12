import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

import keras
import keras.layers.core as core
from keras.models import Sequential, load_model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, GaussianNoise
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras import backend as K
import keras.layers.convolutional as conv

ROWS = 128
COLS = 128
CHANNELS = 3


def open_data_int(beg = 0, end = None):
    TRAIN_DIR = './data/dog_vs_cats/train/'
    train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i][beg:end]
    train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i][beg:end]
    images = train_dogs + train_cats
    data_Y = np.array([[1,0] for _ in range(len(train_dogs))] + [[0,1] for _ in range(len(train_cats))])
    data_X = prep_data_int(images)
    del images
    return data_X, data_Y

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)[...,::-1]


def prep_data_int(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype="uint8")

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if (i+1) % 1000 == 0: print('Processed {} of {}'.format(i+1, count))

    return data

def shuffle_arr(arr1, arr2):
    assert len(arr1) == len(arr2)
    shuffle_index = np.random.choice(len(arr1), len(arr1), replace = False)
    return arr1[shuffle_index], arr2[shuffle_index]

#print("Images shape: {}".format(data_X.shape))

nb_classes = 2

X_train, Y_train = open_data_int(-10000, None)
X_train, Y_train = shuffle_arr(X_train, Y_train)
X_validation, Y_validation = open_data_int(0, 2500)
X_validation, Y_validation = shuffle_arr(X_validation, Y_validation)
#X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size = 0)
"""
for X, Y in zip(X_train, Y_train):
    plt.imshow(X)
    print(Y)
    plt.show()
"""
#print(X_train[0])


def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(pair)
    plt.show()


#optimizer = RMSprop(lr=1e-4)
#objective = 'binary_crossentropy'


def catdog4():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(1, activation="sigmoid"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.binary_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog5():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2, activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog6():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(8, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(8, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2, activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog7():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2, activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog8():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2,   activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog9(): #11
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2, activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

def catdog10():
    model = Sequential()

    #This 3 Layers for 128x128
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.02))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))#, kernel_regularizer=regularizers.l2(0.01),
                    #bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(2, activation="softmax"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=0.0001),#, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model

#model = catdog9()
model = load_model("keras_model_cat_dogs9")

nb_epoch = 100
batch_size = 128


## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

class PrintInfo(Callback):
    def on_batch_end(self, batch, logs):
        print(logs)
        print(batch)
        inp = self.model.input  # input placeholder
        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        # Testing
        test = np.random.random((1,64,64,3))
        layer_outs = [func([test, 1.]) for func in functors]
        #print(layer_outs[-1])
        print(self.params)

def run_catdog():
    history = LossHistory()
    print_info = PrintInfo()
    #model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,
    #          validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2,
              callbacks=[history])
    #score = model.evaluate(X_test, Y_test, verbose=1)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    return history

steps_per_epoch_train = int(np.ceil(len(X_train)/batch_size))
steps_per_epoch_val = int(np.ceil(len(X_validation)/batch_size))
def generator_data(input, targets, num_steps):
    step = 0
    assert len(input) == len(targets)
    len_data = len(input)
    indices = None
    input = input
    targets = targets
    num_steps = num_steps

    while True:
        if step == 0:
            indices = np.random.choice(len_data, len_data, replace = False)
        curr_indices = indices[step*batch_size:(step +1)*batch_size]
        yield input[curr_indices].astype("float32")/255., targets[curr_indices]
        step += 1
        if step == num_steps:
            step = 0
            indices = None

#history = run_catdog()
history = LossHistory()
filepath="models_cat_dogs/keras_model_cat_dogs11-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc')
model.fit_generator(generator=generator_data(X_train, Y_train, steps_per_epoch_train),
                    steps_per_epoch=steps_per_epoch_train, epochs=nb_epoch, verbose=1, callbacks=[history, checkpoint],
                    validation_data=generator_data(X_validation, Y_validation,steps_per_epoch_val),
                    validation_steps=steps_per_epoch_val)

model_num = 11
model.save("keras_model_cat_dogs{}".format(model_num))

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, label = "Validation Loss")
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.savefig("./figures/cats_dogs_model{}.png".format(model_num))
plt.show()

#test_X, test_Y = open_data(11500, None)
score = model.evaluate(test_X, test_Y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
