#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import random

FILE_NAME = 'cullpdb+profile_6133.npy.gz'
NUM_TEST_SET = 5200

def get_data(file_name, num_test_Set):#读取数据
    print('reading data from'+file_name+'...')
    input_datas = np.load(file_name)

    # 6133个samples，每个sample有700个氨基酸
    input_datas.shape = (6133, 700, 57)

    # input_datas.shape = (4293100, 57)
    # print("inputs")
    # print(input_datas[0:5,:5])
    #seqs 氨基酸序列的独热编码
    seqs = input_datas[:, :, :22].astype(np.float32)

    #labels 的独热编码
    labels = input_datas[:, :, 22:31].astype(np.float32)

    #features 氨基酸对应的属性(非独热编码)
    features = input_datas[:, :, 35:].astype(np.float32)

    # print('seq:')
    # print(seqs[1:2,0:10,:])
    # print('labels:')
    # print(labels[1:2, :10, :])
    # print('fetures:')
    # print(features[1:2, 0:2, :])

    #将seq与features合并
    seqs = np.concatenate((seqs, features), axis=2)
    # print(np.shape(x_train))
    # print("x_data:")
    # print(x_train[1:2, :5, 20:24])

    #根据num_test_set将数据集分为训练集和测试集
    train_seqs = seqs[:num_test_Set]
    train_labels = labels[:num_test_Set]
    # train_features = features[:num_test_Set]
    test_seqs = seqs[num_test_Set:]
    test_labels = labels[num_test_Set:]
    # test_features = features[num_test_Set:]

    print('Data processing completed')
    print(np.shape(train_seqs))
    print(np.shape(train_labels))
    print(np.shape(test_seqs))
    print(np.shape(test_labels))

    return train_seqs, train_labels, test_seqs, test_labels

# get_data(FILE_NAME, NUM_TEST_SET)