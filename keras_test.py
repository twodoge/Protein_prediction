#引用所需库
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adadelta

(x_train, y_train), (x_test, y_test) = mnist.load_data()#加载mnist数据

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')#转换格式。(样本数量，长，款，1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')

x_train /= 255#像素大小介于0~255转化为0~1
y_test /= 255

y_train = np_utils.to_categorical(y_train, 10) #生成one-hot编码
y_test = np_utils.to_categorical(y_test, 10)

#构建模型
model = Sequential()
#第一层卷积层
model.add(Conv2D(filters = 64, kernel_size= (3, 3), activation='relu', input_shape= (28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
#第二层卷积层
model.add(Conv2D(filters= 64, kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#铺平当前节点
model.add(Flatten())
#生成全连接层
model.add(Dense(128, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
#定义损失函数学习速率
model.compile(loss = 'categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])