#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from keras.models import Sequential
from keras.models import load_model
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

model = load_model('/media/twodog/linux/model/my_mnist_model.h5')
img1 = skimage.io.imread('/media/twodog/linux/image/12.jpg',as_grey=True)
skimage.io.imshow(img1)
plt.show()

img1 = np.reshape(img1, (1, 28, 28, 1)).astype('float32')
# img1 = (img1 - 255) / 255
proba = model.predict_proba(img1, verbose= 0)
result = model.predict_classes(img1, verbose= 0)

print(proba[0])
print(result[0])