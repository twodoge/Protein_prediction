#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling1D, MaxPool2D
from keras.optimizers import SGD

import data_Processing

FILE_NAME = 'cullpdb+profile_6133.npy.gz'
NUM_TEST_SET = 3000000

def cnn_prediction1():
    #Generate train data and test data
    x_train, y_train, x_test, y_test = data_Processing.get_data(FILE_NAME, NUM_TEST_SET)
    x_train = x_train.reshape(1, 3000000 ,44)
    y_train = y_train.reshape(1, 3000000, 9)

    # print(np.shape(x_train))
    # print("x_data:")
    # print(x_train[1:2, :5, 18:26])
    # print(np.shape(y_train))
    # print("y_train:")
    # print(y_train[1:2, :5, :5])

    print(np.shape(x_train))
    print(np.shape(y_train))

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(3000000, 44)))
    model.add(Dense(9, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=32)

    print("finish")

def base_on_mlp():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    # Generate dummy data
    x_train, y_train, x_test, y_test = data_Processing.get_data(FILE_NAME, NUM_TEST_SET)
    x_train = x_train.reshape(1, 3000000, 44)
    y_train = y_train.reshape(1, 3000000, 9)
    x_test = x_test.reshape(1, 1293100, 44)
    y_test = y_test.reshape(1, 1293100, 9)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_shape=(3000000, 44)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)

if __name__ == "__main__":
    base_on_mlp()