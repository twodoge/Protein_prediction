#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import random

FILE_NAME = 'cullpdb+profile_6133.npy.gz'
TRAIN_SET = 5600
TEST_SET = 5605
VALID_SET = 5877

def get_cnn_data(file_name,train_set, test_set, valid_set):#读取数据
    print('reading data from'+file_name+'...')
    input_datas = np.load(file_name)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    input_datas.shape = (-1, 700, 57)
    # with open("C:\\bishe\data\dssp\data", 'a', encoding='gb18030') as f:
    #     f.write(str(input_datas[0:2, 0:700, :]))

    train_data = input_datas[ : 5600].astype(np.float32)#[0,5600）训练
    test_data = input_datas[5605 : 5877].astype(np.float32)#[5605,5877）=272
    valid_data = input_datas[5877 : 6133].astype(np.float32)#[5877,6133]验证 = 256

    # print("初始数据：")
    # print(valid_data[0, 8, 0:57])
    #训练数据. 20：氨基酸的独热编码，21：feature
    x_train = np.zeros((5600*(700-16), 17, 43))
    seq = np.zeros((17, 22))
    feature = np.zeros((17, 21))
    y_train = np.zeros((5600*(700-16), 9))
    flag = 0
    for i in range(5600):
        for j in range(700-16):
            for k in range(17):
                seq[k, ] = train_data[i, j+k, :22]
                feature[k, ] =train_data[i, j+k, 35:56]
                if k==8:
                    y_train[flag, ] = train_data[i, j + k, 22:31]
            x_train[flag, ] = np.concatenate((seq, feature),axis=1)
            flag = flag + 1

    #测试数据
    x_test = np.zeros((272 * (700 - 16), 17, 43))
    seq = np.zeros((17, 22))
    feature = np.zeros((17, 21))
    y_test = np.zeros((272 * (700 - 16), 9))
    flag = 0
    for i in range(272):
        for j in range(700 - 16):
            for k in range(17):
                seq[k,] = test_data[i, j + k, :22]
                feature[k,] = test_data[i, j + k, 35:56]
                if k == 8:
                    y_test[flag,] = test_data[i, j + k, 22:31]
            x_test[flag,] = np.concatenate((seq, feature), axis=1)
            flag = flag + 1

    #验证数据
    x_valid = np.zeros((256 * (700 - 16), 17, 43))
    seq = np.zeros((17, 22))
    feature = np.zeros((17, 21))
    y_valid = np.zeros((256 * (700 - 16), 9))
    flag = 0
    for i in range(256):
        for j in range(700 - 16):
            for k in range(17):
                seq[k,] = valid_data[i, j + k, :22]
                feature[k,] = valid_data[i, j + k, 35:56]
                if k == 8:
                    y_valid[flag,] = valid_data[i, j + k, 22:31]
            x_valid[flag,] = np.concatenate((seq, feature), axis=1)
            flag = flag + 1

    print("x_train, y_train, x_test, y_test, x_vail, y_vail data shape:")
    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))
    print(np.shape(x_valid))
    print(np.shape(y_valid))
    print('Data processing completed')
    #
    # print(x_valid[0, 8, 0:43])
    # print(y_valid[0, 0:9])


    return x_train, y_train, x_test, y_test, x_valid, y_valid

# get_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
# get_cnn_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)