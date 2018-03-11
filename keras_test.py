#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.preprocessing import image
import skimage.io

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)

plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))

plt.show()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# X_train = X_train.reshape(1, 28, 28, 1).astype('float32') ValueError: cannot reshape array of size 47040000 into shape (1,28,28,1)
# X_test = X_test.reshape(1, 28, 28, 1).astype('float32')  ValueError: cannot reshape array of size 47040000 into shape (1,28,28,1)
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')    <---4
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


###raw
# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    # model.add(Conv2D(30, (5, 5), padding='valid', input_shape=(28, 28,1), activation='relu'))   <----3,2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # optimizer  优化器
    # loss 损失函数
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = larger_model()
# Fit the model
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
          verbose=2)  # epochs 200 too bigger
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=200, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

# save the model
model.save('/media/twodog/linux/model/my_model.h5')  # creates a HDF5 file 'my_model.h5'