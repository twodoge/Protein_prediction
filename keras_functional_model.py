#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)

print(a)
print(b)
model = Model(inputs=1, outputs=b)