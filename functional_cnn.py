#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, GlobalAveragePooling1D, MaxPooling1D, MaxPool2D, MaxPooling2D
from keras.optimizers import SGD
from keras import metrics

def conv2d_fuctional():
    x_train, y_train, x_test, y_test = data_Processing.get_dssp_data()
    x_train = x_train.reshape(-1, 7, 28, 1)
    x_test = x_test.reshape(-1, 7, 28, 1)

    inputs = Input(shape = (7, 28, -1))

    inputs_1 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(inputs)
    inputs_1 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(inputs_1)

    inputs_2 = Conv2D(32, (2, 2), padding = 'same', activation = 'relu')(inputs)
    inputs_2 = Conv2D(32, (5, 5), padding = 'same', activation = 'relu')(inputs_2)

    inputs_3 = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same')(inputs)
    inputs_3 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu')(inputs_3)

    outputs = keras.layers.concatenate([inputs_1, inputs_2, inputs_3], axis = 1)

    outputs = Dense(64, activation = 'relu')(outputs)
    outputs = Dropout((0.2))
    outputs = Dense(8, activation = 'softmax')

    model = Model(inputs = inputs, outputs = outputs)

    sgd = SGD(lr = 0.01, decay= 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizers = sgd, loss = 'categorical_crossentropy', metrics = ['acc'])

    #当验证集的loss不在下降时，中断训练
    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 3)

    his = model.fix(x_train, y_train, batch_size = 256, epoch = 20, verbose = 1,
        validation_split = 0.2,
        callbacks = [earlyStopping],
        shuffle = True,
        )

    print(his)
    model.summary()
    with open('C:\\bishe\model\history_funtional_cnn','w') as f:
        f.write(his)

    #保存模型
    model.save_weights('c:\\bishe\model\cnn_2d_weights_funtional.h5')
    model.save('c:\\bishe\model\cnn_2d_model_funtional.h5')

    #画出模型结构图，保存到图片
    from keras.utils import plot_model
    plot_model(model, to_file = 'con_2d_model_fun.PNG', show_shapes = True)

if __name__ == "__main__":
    conv2d_fuctional()


