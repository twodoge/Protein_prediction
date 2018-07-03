#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import keras
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten,LSTM,Bidirectional, Activation
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling1D, MaxPool2D, MaxPooling2D
from keras.layers.core import Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD
from keras import metrics
import data_Processing
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig('C:\\bishe\model\cnn_2d_model_922_funtionalkkk.png')

#除去Noseq后的预测精确度
def acc_Q8(y_true, y_pred):
    correct = 0
    count = np.size(y_true, axis=0)
    total = count
    print(np.shape(y_true),np.shape(y_pred))
    # print("total:",total)
    # print(y_true[0:3, 0:9])
    # print(y_pred[0:3, 0:9])
    for i in range(count):
        if y_true[i , 8] == 1:
            total -= 1
        else:
            if y_true[i, np.argmax(y_pred[i, :])] == 1:
                correct += 1

    # print("total:", total, "correct:", correct)
    return correct/total

    #除去Noseq后的预测精确度
def acc_Q3(y_true, y_pred):
    correct = 0
    count = np.size(y_true, axis=0)
    total = count
    print(np.shape(y_true),np.shape(y_pred))
    # print("total:",total)
    # print(y_true[0:3, 0:9])
    # print(y_pred[0:3, 0:9])
    for i in range(count):
        if y_true[i , 3] == 1:
            total -= 1
        else:
            if y_true[i, np.argmax(y_pred[i, :])] == 1:
                correct += 1

    # print("total:", total, "correct:", correct)
    return correct/total

#输出
def print_Q8(y_true, y_pred):
    #Noseq = Z
    structs = 'ACEDGFIHKMLNQPSRTWVYXZ'
    correct = 0
    count = np.size(y_true, axis=0)
    total = count
    # print(np.shape(y_true),np.shape(y_pred))
    # print("total:",total)
    pred = []
    true = []
    for i in range(count):
        pred.append(structs[np.argmax(y_pred[i, :])])
        true.append(structs[np.argmax(y_true[i, :])])

    print("pred:", pred[0:20])
    print("true:", true[0:20])
    # print("total:", total, "correct:", correct)
    return correct/total

#混淆矩阵
def mix_matrix(y_true, y_pred):
    mix_matrix = np.zeros((8, 9))
    count =0
    for i in range(np.size(y_true, axis = 0)):
        for j in range(8):
            if y_true[i, j] == 1:
                mix_matrix[j, 0] += 1
                flag = j
                count += 1
                break
        if np.argmax(y_pred[i, :]) < 8:
            temp = np.argmax(y_pred[i, :]) +1
            mix_matrix[flag, temp] += 1
    print(count)
    print('混淆矩阵：')
    for i in range(8):
        print(mix_matrix[i,])

    for i in range(8):
        for j in range(1, 9):
            mix_matrix[i,j] = mix_matrix[i, j] / mix_matrix[i, 0]
    print('混淆矩阵：')
    for i in range(8):
        print(mix_matrix[i,])

        #混淆矩阵
def mix_3matrix(y_true, y_pred):
    mix_matrix = np.zeros((4, 5))
    count =0
    for i in range(np.size(y_true, axis = 0)):
        for j in range(4):
            if y_true[i, j] == 1:
                mix_matrix[j, 0] += 1
                flag = j
                count += 1
                break
        if np.argmax(y_pred[i, :]) < 4:
            temp = np.argmax(y_pred[i, :]) +1
            mix_matrix[flag, temp] += 1
    print(count)
    print('混淆矩阵：')
    for i in range(4):
        print(mix_matrix[i,])

    for i in range(4):
        for j in range(1, 5):
            mix_matrix[i,j] = mix_matrix[i, j] / mix_matrix[i, 0]
    print('混淆矩阵：')
    for i in range(4):
        print(mix_matrix[i,])

#Q3分类
def Q3conv2d_fuctional():
    window_size = 9
    time = datetime.datetime.now().strftime('%m%d_%H:%M')
    x_train, y_train = data_Processing.get_cb513_3train_data(window_size)
    x_test, y_test = data_Processing.get_cb513_3test_data(window_size)

    x_train = x_train.reshape(-1, np.size(x_train, axis=1), np.size(x_train, axis=2), 1)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    print(np.shape(x_train),np.shape(y_train))

    inputs = Input(shape = ( np.size(x_train, axis=1), np.size(x_train, axis=2), 1))  

    inputs_1 = Conv2D(64, (2, 2), padding = 'same', activation = 'relu')(inputs)  
    inputs_1 = MaxPooling2D((2, 2), strides = (1, 1), padding = 'same')(inputs_1)  
    inputs_1 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_1)  
    inputs_1 = Dropout((0.1))(inputs_1)  
    
    inputs_2 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(inputs)  
    inputs_2 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs_2)  
    inputs_2 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_2)  
    inputs_2 = Dropout((0.1))(inputs_2)  
    
    inputs_3 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs)  
    inputs_3 = Conv2D(32, (1, 1), padding = 'same', activation = 'relu')(inputs_3)  
    inputs_3 = Dropout((0.2))(inputs_3)  

    outputs = keras.layers.concatenate([inputs_1, inputs_2, inputs_3], axis = 1)

    outputs = Flatten()(outputs)
    outputs = Dense(512, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(128, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(4, activation = 'softmax')(outputs)

    model = Model(inputs = inputs, outputs = outputs)
    # model = load_model("C:\\bishe\model\cnn_2d_model_9_44_funtional_.h5")

    sgd = SGD(lr = 0.1, decay= 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])

    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

    history = model.fit(x_train, y_train, batch_size = 128, epochs = 1, verbose = 1,
        validation_split = 0.1,
        callbacks = [earlyStopping],
        shuffle = True,
        )
        # call the function
    # training_vis(history)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print(score)
    # model.summary()

    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=128, verbose=1)
    # result = acc_Q8(y_test, prediction)
    # print("acc_Q8:",result)
    print(prediction)

    #保存模型
    save_model = "C:\\bishe\model\cnn_2d_model_9_44_3new_funtional_.h5"
    # model.save_weights('c:\\bishe\model\cnn_2d_weights_funtional.h5')
    model.save(save_model)

    #画出模型结构图，保存到图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\model\cnn_2d_model_3_funtional.png'
    plot_model(model, to_file = png_f, show_shapes = True)
    # for i in history:
    #     print(i)
    # mix_matrix(y_test, prediction)

def Q8conv2d_fuctional():
    window_size = 9
    time = datetime.datetime.now().strftime('%m%d_%H:%M')
    x_train, y_train = data_Processing.get_cb513_train_data(window_size)
    x_test, y_test = data_Processing.get_cb513_test_data(window_size)

    x_train = x_train.reshape(-1, np.size(x_train, axis=1), np.size(x_train, axis=2), 1)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    print(np.shape(x_train),np.shape(y_train))

    inputs = Input(shape = ( np.size(x_train, axis=1), np.size(x_train, axis=2), 1))  

    inputs_1 = Conv2D(64, (2, 2), padding = 'same', activation = 'relu')(inputs)  
    inputs_1 = MaxPooling2D((2, 2), strides = (1, 1), padding = 'same')(inputs_1)  
    inputs_1 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_1)  
    inputs_1 = Dropout((0.1))(inputs_1)  
    
    inputs_2 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(inputs)  
    inputs_2 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs_2)  
    inputs_2 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_2)  
    inputs_2 = Dropout((0.1))(inputs_2)  
    
    inputs_3 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs)  
    inputs_3 = Conv2D(32, (1, 1), padding = 'same', activation = 'relu')(inputs_3)  
    inputs_3 = Dropout((0.2))(inputs_3)  

    outputs = keras.layers.concatenate([inputs_1, inputs_2, inputs_3], axis = 1)

    outputs = Flatten()(outputs)
    outputs = Dense(2048, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(512, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(128, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(9, activation = 'softmax')(outputs)

    model = Model(inputs = inputs, outputs = outputs)
    # model = load_model("C:\\bishe\model\cnn_2d_model_9_44_funtional_.h5")

    sgd = SGD(lr = 0.01, decay= 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])

    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

    history = model.fit(x_train, y_train, batch_size = 256, epochs = 1, verbose = 1,
        validation_split = 0.1,
        callbacks = [earlyStopping],
        shuffle = True,
        )
        # call the function
    # training_vis(history)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print(score)
    # model.summary()

    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=128, verbose=1)
    result = acc_Q8(y_test, prediction)
    print("acc_Q8:",result)
    # print(prediction)

    #保存模型
    save_model = "C:\\bishe\model\cnn_2d_model_944new_funtional.h5"
    # model.save_weights('c:\\bishe\model\cnn_2d_weights_funtional.h5')
    model.save(save_model)

    #画出模型结构图，保存到图片
    # from keras.utils import plot_model
    # png_f = 'C:\\bishe\model\cnn_2d_model_3_funtional.png'
    # plot_model(model, to_file = png_f, show_shapes = True)
    # for i in history:
    #     print(i)
    mix_matrix(y_test, prediction)

def conv2d_lstm():
    window_size = 9
    time = datetime.datetime.now().strftime('%m%d_%H:%M')
    x_train, y_train = data_Processing.get_cb513_train_data(window_size)
    x_test, y_test = data_Processing.get_cb513_test_data(window_size)

    x_train = x_train.reshape(-1, np.size(x_train, axis=1), np.size(x_train, axis=2), 1)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    print(np.shape(x_train),np.shape(y_train))

    inputs = Input(shape = ( np.size(x_train, axis=1), np.size(x_train, axis=2), 1))  

    inputs_1 = Conv2D(64, (2, 2), padding = 'same', activation = 'relu')(inputs)  
    inputs_1 = MaxPooling2D((2, 2), strides = (1, 1), padding = 'same')(inputs_1)  
    inputs_1 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_1)  
    inputs_1 = Dropout((0.4))(inputs_1)  
    
    inputs_2 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(inputs)  
    inputs_2 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs_2)  
    inputs_2 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs_2)  
    inputs_2 = Dropout((0.4))(inputs_2)  
    
    inputs_3 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs)  
    inputs_3 = Conv2D(32, (1, 1), padding = 'same', activation = 'relu')(inputs_3)  
    inputs_3 = Dropout((0.2))(inputs_3)  

    outputs = keras.layers.concatenate([inputs_1, inputs_2, inputs_3], axis = 1)
    print(np.shape(outputs))

    outputs = TimeDistributed(Flatten())(outputs)
    outputs = TimeDistributed(Dense(64))(outputs)
    outputs = Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2,return_sequences = True))(outputs)
    outputs = LSTM(64)(outputs)
    outputs = Dropout((0.5))(outputs)

    outputs = Dense(128, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(9, activation = 'softmax')(outputs)

    model = Model(inputs = inputs, outputs = outputs)
    # model = load_model("C:\\bishe\model\cnn_2d_model_9_44_funtional_.h5")

    sgd = SGD(lr = 0.1, decay= 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])

    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

    history = model.fit(x_train, y_train, batch_size = 256, epochs = 1, verbose = 1,
        validation_split = 0.2,
        callbacks = [earlyStopping],
        shuffle = True,
        )
        # call the function
    # training_vis(history)
    score = model.evaluate(x_test, y_test, batch_size=256, verbose=1)
    print(score)
    # model.summary()

    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=128, verbose=1)
    result = acc_Q8(y_test, prediction)
    print("acc_Q8:",result)

    #保存模型
    save_model = "C:\\bishe\model\cnn_2d_lstm.h5"
    # model.save_weights('c:\\bishe\model\cnn_2d_weights_funtional.h5')
    model.save(save_model)

    #画出模型结构图，保存到图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\model\cnn_2d_lstm.png'
    # plot_model(model, to_file = png_f, show_shapes = True)
    # for i in history:
    #     print(i)
    mix_matrix(y_test, prediction)

def conv2d_lstm_fuctional():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2,return_sequences = True),input_shape=(9, 44)))
    model.add(LSTM(64,dropout=0.5))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(9,use_bias=True))
    model.add(Activation("sigmoid"))

    #画出模型结构图，保存到图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\model\lstm.png'
    plot_model(model, to_file = png_f, show_shapes = True)

def conv2d():
    window_size = 9
    x_train, y_train = data_Processing.get_cb513_train_data(window_size)
    x_test, y_test = data_Processing.get_cb513_test_data(window_size)

    x_train = x_train.reshape(-1, np.size(x_train, axis=1), np.size(x_train, axis=2), 1)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    print(np.shape(x_train),np.shape(y_train))

    inputs = Input(shape = ( np.size(x_train, axis=1), np.size(x_train, axis=2), 1))  

    inputs_1 = Conv2D(64, (2, 2), padding = 'same', activation = 'relu')(inputs)
    inputs_1 = MaxPooling2D((2, 2), strides = (1, 1), padding = 'same')(inputs_1)
    outputs = Dropout((0.4))(inputs_1)

    outputs = Flatten()(outputs)
    outputs = Dense(512, activation = 'relu')(outputs)
    outputs = Dropout((0.2))(outputs)
    outputs = Dense(9, activation = 'softmax')(outputs)

    model = Model(inputs = inputs, outputs = outputs)
    # model = load_model("C:\\bishe\model\cnn_2d_model_9_44_funtional_.h5")

    sgd = SGD(lr = 0.01, decay= 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])

    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 2)

    history = model.fit(x_train, y_train, batch_size = 128, epochs = 200, verbose = 1,
        validation_split = 0.2,
        callbacks = [earlyStopping],
        shuffle = True,
        )
        # call the function
    training_vis(history)
    print(history)
    score = model.evaluate(x_test, y_test, batch_size=256, verbose=1)
    print(score)
    # model.summary()

    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=128, verbose=1)
    result = acc_Q8(y_test, prediction)
    print("acc_Q8:",result)

    #保存模型
    save_model = "C:\\bishe\model\cnn_2d_model.h5"
    # model.save_weights('c:\\bishe\model\cnn_2d_weights_funtional.h5')
    model.save(save_model)

    #画出模型结构图，保存到图片
    from keras.utils import plot_model
    png_f = 'C:\\bishe\model\cnn_2d_model.png'
    # plot_model(model, to_file = png_f, show_shapes = True)
    mix_matrix(y_test, prediction)

def prediction():
    window_size = 17

    x_test, y_test = data_Processing.get_cb513_test_data(window_size)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    print(np.shape(x_test),np.shape(y_test))

    model = load_model("C:\\bishe\model\cnn_2d_model_9_44_funtional_.h5")
    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=64, verbose=1)
    result = acc_Q8(y_test, prediction)
    print("acc_Q8:",result)

    mix_matrix(y_test, prediction)

def print_prediction():
    window_size = 9

    x_test, y_test = data_Processing.get_cb513_test_data(window_size)
    x_test = x_test.reshape(-1, np.size(x_test, axis=1), np.size(x_test, axis=2), 1)
    # print(np.shape(x_test),np.shape(y_test))

    model = load_model("C:\\bishe\model\cnn_2d_model_9_44_1_funtional_.h5")
    #评估自己的模型，即删除Noseq后的预测准确度
    prediction = model.predict(x_test, batch_size=16, verbose=1)
    result2 = acc_Q8(y_test, prediction)
    print("acc_Q8:",result2)
    # mix_3matrix(y_test, prediction)
    result1 = print_Q8(y_test, prediction)

if __name__ == "__main__":
    prediction()
    # conv2d_fuctional()
    # conv2d_lstm_fuctional()
    # conv2d()
    # conv2d_lstm()
    # print_prediction()
    # Q8conv2d_fuctional()
