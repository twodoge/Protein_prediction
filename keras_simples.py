#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import numpy as np
import keras

def simple_classification():
    #For a single input model with 10 classification
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=1000))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    #Generate dummy data
    data = np.random.random((10000, 1000))
    labels = np.random.randint(10, size=(10000, 1))

    #Convert labels to categorical one-hot encoding
    one_hot_labels = K.utils.to_categorical(labels, num_classes=10)

    #Train the model, iterating on the data in batchs of 32 samples
    model.fit(data, one_hot_labels, epochs=20, batch_size=64)

    #Save model
    model.save('simple_classification.h5')
    model.save_weights('simple_calssification_weights.h5')

#基于多层感知机的softmax多分类
def base_on_softmax_classification():
    #Generate dummy data
    train_data = np.random.random((1000,20))
    train_labels = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    test_data = np.random.random((100,20))
    test_labels = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    #Built model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #Gradient optimization
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )

    model.fit(train_data, train_labels,
              epochs=20,
              batch_size=128
              )

    score = model.evaluate(test_data, test_labels, batch_size=128)
    print(score)

def VGG_CNN():
    # Generate dummy data
    x_train = np.random.random((100, 100, 100, 3))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((20, 100, 100, 3))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)

def self_data():
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.optimizers import Adam
    import numpy as np

    model = Sequential()
    model.add(Dense(32, input_shape=(5, 3)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(4))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam)

    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    x = x.reshape(1, 5, 3)

    y = model.predict(x)
    print(y)

if __name__ == "__main__":
    self_data()