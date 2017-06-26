import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

import keras
import keras.layers.core as core
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, GaussianNoise
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import keras.layers.convolutional as conv

ROWS = 64
COLS = 64
CHANNELS = 3


def open_data(beg = 0, end = None):
    TRAIN_DIR = './data/dog_vs_cats/train/'
    train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i][beg:end]
    train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i][beg:end]
    images = train_dogs + train_cats
    data_Y = np.array([1 for _ in range(len(train_dogs))] + [0 for _ in range(len(train_cats))])
    data_X = prep_data(images) / 255.
    del images
    return data_X, data_Y

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)[...,::-1]


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i % 1000 == 0: print('Processed {} of {}'.format(i, count))

    return data



#print("Images shape: {}".format(data_X.shape))

nb_classes = 2

data_X, data_Y = open_data(0, 11500)
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size = 0)
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


def catdog():
    model = Sequential()


    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    #model.add(GaussianNoise(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.1))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.1))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(GaussianNoise(0.1))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    #model.add(core.Dense(nb_classes, activation="softmax"))
    model.add(Dense(1, activation="sigmoid"))#, kernel_initializer='zeros'))

    model.compile(loss=keras.losses.binary_crossentropy,
        optimizer = keras.optimizers.Adam(),#lr=0.001, decay=3e-4),
        metrics = ['accuracy'])
    model.summary()
    return model


model = catdog()
#model = create_model_kaggle()

nb_epoch = 50
batch_size = 128


## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

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


history = run_catdog()

model.save("keras_model_cat_dogs4")

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, label = "Validation Loss")
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()

test_X, test_Y = open_data(11500, None)
score = model.evaluate(test_X, test_Y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
