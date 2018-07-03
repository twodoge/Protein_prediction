#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling1D, MaxPool2D, MaxPooling2D
from keras.optimizers import SGD
from keras import metrics
import datetime
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt

import data_Processing

FILE_NAME = 'C:\\bishe\data\\available\cullpdb+profile_6133.npy.gz'
TRAIN_SET = 5600
TEST_SET = 5605
VALID_SET = 5877

def cnn_prediction1():
    #Generate train data and test data
    x_train, y_train, x_test, y_test, x_vail, y_vail = data_Processing.get_data(FILE_NAME, NUM_TEST_SET)
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

    # x_train, y_train, x_test, y_test , x_valid, y_valid= data_Processing.get_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
    # x_train = x_train.reshape(-1, 700, 44)
    # y_train = y_train.reshape(-1, 700, 9)
    # x_test = x_test.reshape(-1, 700, 48)
    # y_test = y_test.reshape(-1, 700, 9)
    x_train, y_train, x_test, y_test = data_Processing.get_seq_data()
    y_train = y_train.reshape(-1, 8, 8)
    y_test = y_test.reshape(-1, 8, )

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(7, 28)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    #validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。
    model.fit(x_train, y_train,
              epochs=1,
              batch_size=32,
              verbose=1,
              # validation_data=(x_valid, y_valid)
              validation_split = 0.1
              )
    score = model.evaluate(x_test, y_test, batch_size=128)
    model.summary()
    print(score)
    #score print out [0.45577393547160794, 0.8245454501600677]
    #Save model
    model.save("C:\\bishe\model\mlp_7_funtional_.h5")
    # model.save_weights('base_on_mlp_model_weights.h5')
        #画出模型结构图，并保存成图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\data\\available\mlp_7_model.png'
    plot_model(model, to_file= png_f, show_shapes=True)

def CNN_1D():
    # Generate dummy data
    # x_train, y_train, x_test, y_test = data_Processing.get_data(FILE_NAME, NUM_TEST_SET, MUN_TEST_SET, NUM_VALI_SET)
    # x_train = x_train.reshape(-1, 700, 44)
    # y_train = y_train.reshape(-1, 700, 9)
    # x_test = x_test.reshape(-1, 700, 44)
    # y_test = y_test.reshape(-1, 700, 9)

    x_train, y_train, x_test, y_test = data_Processing.get_seq_data()


    # x_train = np.reshape(x_train, (-1, 44, 1))
    # y_train = np.reshape(y_train, (-1, 9))
    # x_test = np.reshape(x_test, (-1, 44, 1))
    # y_test = np.reshape(y_test, (-1, 9))

    # x_train = np.expand_dims(x_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    # x_test = np.expand_dims(x_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    model = Sequential()
    # 第一个卷积层，64个卷积核，每个卷积核大小为3
    model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=(7, 28)))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    # model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.05)
    score = model.evaluate(x_test, y_test, batch_size=128)

    model.summary()
    print(score)
    #score print out [0.42530638596938725, 0.8390732568402471]

def mlp():
    x_train, y_train, x_test, y_test , x_valid, y_valid= data_Processing.get_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
    x_train = x_train.reshape(-1, 42)
    y_train = y_train.reshape(-1, 8)
    x_test = x_test.reshape(-1, 42)
    y_test = y_test.reshape(-1, 8)
    x_valid =x_valid.reshape(-1, 42)
    y_valid = y_valid.reshape(-1, 8)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=42))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=100,
              batch_size=64,
              validation_data=(x_valid, y_valid)
              )
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.summary()
    print(score)
    # score print out
    # Save model
    model.save('base_on_mlp_model.h5')
    model.save_weights('base_on_mlp_model_weights.h5')

def acc_8(y_true, y_pred):
    correct = 0
    count = np.size(y_true, axis=0)
    total = count
    print(np.shape(y_true),np.shape(y_pred))
    print("total:",total, 'll:', np.size(y_pred, axis=1))
    print(y_pred[0:10, 0:9])
    for i in range(count):
        if y_pred[i , 8] == 1:
            total -= 1
        else:
            if y_true[i, np.argmax(y_pred[i, :])] == 1:
                    correct += 1

    print("total:", total, "correct:", correct)
    return correct/total

def cnn_1d():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    x_train, y_train, x_test, y_test, x_valid, y_valid = data_Processing.get_cnn_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
    x_train = x_train.reshape(-1, 43, 17)
    y_train = y_train.reshape(-1, 9)
    x_test = x_test.reshape(-1, 43, 17)
    y_test = y_test.reshape(-1, 9)
    x_valid =x_valid.reshape(-1, 43, 17)
    y_valid = y_valid.reshape(-1, 9)

    model = Sequential()
    model.add(Conv1D(64, 3, input_shape=(np.size(x_train, axis=1), np.size(x_train, axis=2))))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)
    score = model.evaluate(x_test, y_test, batch_size=64)
    print(score)
    model.summary()

    prediction = model.predict(x_valid, batch_size=64, verbose=1)
    print("acc_8:", acc_8(y_valid, prediction))

# define the function 画出曲线
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
#除去Noseq后的预测精确度
def acc_Q8(y_true, y_pred):
    correct = 0
    count = np.size(y_true, axis=0)
    total = count
    print(np.shape(y_true),np.shape(y_pred))
    print("total:",total)
    print(y_true[0:3, 0:9])
    print(y_pred[0:3, 0:9])
    for i in range(count):
        if y_true[i , 8] == 1:
            total -= 1
        else:
            if y_true[i, np.argmax(y_pred[i, :])] == 1:
                correct += 1

    print("total:", total, "correct:", correct)
    return correct/total

def cnn2d():
    window_size = 9
    time = datetime.datetime.now().strftime('%m%d_%H:%M')
    x_train, y_train = data_Processing.get_cb513_train_data(window_size)
    x_test, y_test = data_Processing.get_cb513_test_data(window_size)

    x_train = x_train.reshape(-1, np.size(x_train, axis=1), np.size(x_train, axis=2), 1)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=( np.size(x_train, axis=1), np.size(x_train, axis=2), 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['acc'])
    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

    his = model.fit(x_train, y_train, batch_size = 128, epochs = 1, verbose = 1,
        validation_split = 0.1,
        callbacks = [earlyStopping],
        shuffle = True,
        )

    # print(his.history)

    score = model.evaluate(x_test, y_test, batch_size=128)
    print(score)
    model.summary()

    # call the function
    training_vis(his)

    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=128, verbose=1)
    result = acc_Q8(y_test, prediction)
    print("acc_Q8:",result)

    #保存模型
    # model.save_weights("C:\\bishe\data\\available\cnn2d_weights_"+time+".h5")
    model.save("C:\\bishe\data\\available\cnn2d_model_.h5")

    # #画出模型结构图，并保存成图片
    # from keras.utils import plot_model
    # plot_model(model, to_file='C:\\bishe\data\\available\cnn2d_model_'+time+'.png', show_shapes=True)

def cnn2d_dssp():
    x_train, y_train, x_test, y_test = data_Processing.get_seq_data()
    x_train = x_train.reshape(-1, 7, 28, 1)
    # y_train = y_train.reshape(-1, 8)
    x_test = x_test.reshape(-1, 7, 28, 1)
    # y_test = y_test.reshape(-1, 8)
    # x_valid =x_valid.reshape(-1, 17, 43, 1)
    # y_valid = y_valid.reshape(-1, 8)
    time = datetime.datetime.now().strftime('%m%d_%H:%M')

    model = Sequential()
    # model = load_model('C:\\bishe\model')
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(7, 28, 1)))
    # model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['acc'])

    #当验证集的loss不再下降时，中断训练
    from keras.callbacks import EarlyStopping
    early_stoppping = EarlyStopping(monitor='val_loss', patience=2)

    his = model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1,
              validation_split=0.1,
              callbacks = [early_stoppping],
              shuffle = True,
              )
    print(his.history)

    # score = model.evaluate(x_test, y_test, batch_size=128)
    # print(score)
    model.summary()

    #评估自己的模型，即删除Noseq
    # prediction = model.predict(x_test, batch_size=64, verbose=1)
    # print(prediction)
    score = model.evaluate(x_test, y_test, batch_size=64)
    print(score)

    #保存模型
    # model.save_weights()
    save_model = "C:\\bishe\data\\available\cnn2d_dssp_926_weights.h5"
    model.save(save_model)

    #画出模型结构图，并保存成图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\data\\available\cnn2d_dssp_926_model.png'
    plot_model(model, to_file= png_f, show_shapes=True)

if __name__ == "__main__":
    cnn2d()