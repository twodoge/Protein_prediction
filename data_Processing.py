#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import random
import sqlite3 as sql

FILE_NAME = 'C:\\bishe\data\\available\cullpdb+profile_6133.npy.gz'
TRAIN_SET = 5600
TEST_SET = 5605
VALID_SET = 5877

'''
57个功能是：
“[0,22]：氨基酸残基，顺序为'A'，'C'，'E'，'D'，'G'，'F'，'I'，'H'，'K' ，'M'，'L'，'N'，'Q'，'P'，'S'，'R'，'T'，'W'，'V'，'Y'，'X'，' NoSeq'”
“[22,31]：二级结构标签，具有'L'，'B'，'E'，'G'，'I'，'H'，'S'，'T'，'NoSeq' “
“[31,33]：N-和C-终端;”
“[33,35]：相对和绝对溶剂可及性，仅用于训练（绝对可达性定为15;相对可及性通过蛋白质中最大的可达性值归一化，阈值为0.15;原始溶剂可及性由DSSP计算）”
“[35,57]：sequence profile。注意氨基酸残基的顺序是ACDEFGHIKLMNPQRSTVWXY，它与氨基酸残基”
'''
def get_cb513_train_data(window_size):#读取数据
    train_file = 'C:\\bishe\data\\available\cullpdb+profile_6133_filtered.npy.gz'
    print('reading data...')
    train_data = np.load(train_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    train_data.shape = (-1, 700, 57)
    train_data = train_data[ : ].astype(np.float32)

    # count = 0#统计Noseq数量
    # size = np.size(train_data, axis=0)
    # for i in range(size):
    #     for j in range(700):
    #         if train_data[i, j, 30] == 1:
    #             count += 1
    # print('count:', count)

    #滑动窗口技术
    size = np.size(train_data, axis=0)*(700 - window_size +1)
    x_train = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_train = np.zeros((size, 9))
    #训练集
    flag = 0
    for i in range(np.size(train_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = train_data[i, j+k, :22]
                feature[k, ] = train_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    y_train[flag, ] = train_data[i, j+k, 22:31]
            x_train[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1

    print("x_train, y_train data shape:")
    print(np.shape(x_train))
    print(np.shape(y_train))
    print('train Data processing completed')

    return x_train, y_train

def get_cb513_test_data(window_size):#读取数据
    test_file = 'C:\\bishe\data\\available\cb513+profile_split1.npy.gz'
    print('reading data...')
    test_data = np.load(test_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    test_data.shape = (-1, 700, 57)
    test_data = test_data[ : ].astype(np.float32)

    #测试集
    size = np.size(test_data, axis=0)*(700 - window_size +1)
    x_test = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_test = np.zeros((size, 9))

    flag = 0
    for i in range(np.size(test_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = test_data[i, j+k, :22]
                feature[k, ] = test_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    y_test[flag, ] = test_data[i, j+k, 22:31]
            x_test[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1

    count = 0
    size = np.size(y_test, axis=0)
    for i in range(size):
        if y_test[i, 8] == 1:
            count += 1
    print('count:', count)


    print("x_test, y_test data shape:")
    print(np.shape(x_test))
    print(np.shape(y_test))
    print('Data processing completed')

    return x_test, y_test

def get_cb513_3train_data(window_size):#读取数据
    train_file = 'C:\\bishe\data\\available\cullpdb+profile_6133_filtered.npy.gz'
    print('reading data...')
    train_data = np.load(train_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    train_data.shape = (-1, 700, 57)
    train_data = train_data[ : ].astype(np.float32)

    # count = 0#统计Noseq数量
    # size = np.size(train_data, axis=0)
    # for i in range(size):
    #     for j in range(700):
    #         if train_data[i, j, 30] == 1:
    #             count += 1
    # print('count:', count)

    #滑动窗口技术
    size = np.size(train_data, axis=0)*(700 - window_size +1)
    x_train = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_train = np.zeros((size, 4))
    #训练集
    flag = 0
    for i in range(np.size(train_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = train_data[i, j+k, :22]
                feature[k, ] = train_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    temp = np.argmax(train_data[i, j+k, 22:31])
                    # print(temp)
                    if temp == 3 or temp == 5:
                        label =[1, 0, 0, 0]
                        y_train[flag, ] = [1, 0, 0, 0]
                        # print( y_test[flag, ])
                        # print(temp)
                    if temp == 1 or temp == 2:
                        label =[0, 1, 0, 0]
                        y_train[flag, ] = [0, 1, 0, 0]
                        # print( y_test[flag, ])
                        # print(temp)
                    if temp == 8:
                        label =[0, 0, 0, 1]
                        y_train[flag, ] = [0, 0, 0, 1]
                    if temp == 0 or temp == 4 or temp == 6 or temp == 7:
                        label =[0, 0, 1, 0]
                        y_train[flag, ] = [0, 0, 1, 0]
            x_train[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1

    print("x_train, y_train data shape:")
    print(np.shape(x_train))
    print(np.shape(y_train))
    print('train Data processing completed')

    return x_train, y_train

def get_cb513_3test_data(window_size):#读取数据'L'0，'B'1，'E'2，'G'3，'I'4，'H'5，'S'6，'T'7，'NoSeq'8
    #H,G归为螺旋(H)、E,B归为折叠（E）、其余结构：I,T,S,_L归为无规则卷曲（C）
    test_file = 'C:\\bishe\data\\available\cb513+profile_split1.npy.gz'
    print('reading data...')
    test_data = np.load(test_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    test_data.shape = (-1, 700, 57)
    test_data = test_data[ : ].astype(np.float32)

    # print(test_data[0,0:10,22:31])

    #测试集
    size = np.size(test_data, axis=0)*(700 - window_size +1)
    x_test = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_test = np.zeros((size, 4))

    flag = 0
    for i in range(np.size(test_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = test_data[i, j+k, :22]
                feature[k, ] = test_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    temp = np.argmax(test_data[i, j+k, 22:31])
                    # print(temp)
                    if temp == 3 or temp == 5:
                        label =[1, 0, 0, 0]
                        y_test[flag, ] = [1, 0, 0, 0]
                        # print( y_test[flag, ])
                        # print(temp)
                    if temp == 1 or temp == 2:
                        label =[0, 1, 0, 0]
                        y_test[flag, ] = [0, 1, 0, 0]
                        # print( y_test[flag, ])
                        # print(temp)
                    if temp == 8:
                        label =[0, 0, 0, 1]
                        y_test[flag, ] = [0, 0, 0, 1]
                    if temp == 0 or temp == 4 or temp == 6 or temp == 7:
                        label =[0, 0, 1, 0]
                        y_test[flag, ] = [0, 0, 1, 0]
                    
            x_test[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1

    count = 0
    # size = np.size(y_test, axis=0)
    # for i in range(size):
    #     if y_test[i, 3] == 1:
    #         count += 1
    # print('count:', count)
    # print(y_test[0:100,:])

    print("x_test, y_test data shape:")
    print(np.shape(x_test))
    print(np.shape(y_test))
    print('Data processing completed')

    return x_test, y_test

def Nofeature_cb513_test_data(window_size):#读取数据
    test_file = 'C:\\bishe\data\\available\cb513+profile_split1.npy.gz'
    print('reading data...')
    test_data = np.load(test_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    test_data.shape = (-1, 700, 57)
    test_data = test_data[ : 10].astype(np.float32)

    #测试集
    size = np.size(test_data, axis=0)*(700 - window_size +1)
    x_test = np.zeros((size, window_size, 22))#22氨基酸独热编码
    seq = np.zeros((window_size, 22))
    y_test = np.zeros((size, 9))

    flag = 0
    for i in range(np.size(test_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = test_data[i, j+k, :22]
                if k == int(window_size/2):
                    y_test[flag, ] = test_data[i, j+k, 22:31]
            x_test[flag, ] = seq
            flag += 1

    count = 0
    size = np.size(y_test, axis=0)
    for i in range(size):
        if y_test[i, 8] == 1:
            count += 1
    print('count:', count)


    print("x_test, y_test data shape:")
    print(np.shape(x_test))
    print(np.shape(y_test))
    print('Data processing completed')

    return x_test, y_test

def Nofeature_cb513_train_data(window_size):#读取数据
    train_file = 'C:\\bishe\data\\available\cullpdb+profile_6133_filtered.npy.gz'
    print('reading data...')
    train_data = np.load(train_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    train_data.shape = (-1, 700, 57)
    train_data = train_data[ : ].astype(np.float32)

    # count = 0#统计Noseq数量
    # size = np.size(train_data, axis=0)
    # for i in range(size):
    #     for j in range(700):
    #         if train_data[i, j, 30] == 1:
    #             count += 1
    # print('count:', count)

    #滑动窗口技术
    size = np.size(train_data, axis=0)*(700 - window_size +1)
    x_train = np.zeros((size, window_size, 22))#22氨基酸独热编码
    seq = np.zeros((window_size, 22))
    y_train = np.zeros((size, 9))
    #训练集
    flag = 0
    for i in range(np.size(train_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = train_data[i, j+k, :22]
                if k == int(window_size/2):
                    y_train[flag, ] = train_data[i, j+k, 22:31]
            x_train[flag, ] = seq
            flag += 1

    print("x_train, y_train data shape:")
    print(np.shape(x_train))
    print(np.shape(y_train))
    print('train Data processing completed')

    return x_train, y_train

def get_cb513_data_Noseq(window_size):#读取数据
    train_file = 'C:\\bishe\data\\available\cullpdb+profile_6133_filtered.npy.gz'
    test_file = 'C:\\bishe\data\\available\cb513+profile_split1.npy.gz'
    print('reading data...')
    train_data = np.load(train_file)
    test_data = np.load(test_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    train_data.shape = (-1, 700, 57)
    test_data.shape = (-1, 700, 57)
    train_data = train_data[ : ].astype(np.float32)
    test_data = test_data[ : ].astype(np.float32)

    # count = 0
    # size = np.size(train_data, axis=0)
    # for i in range(size):
    #     for j in range(700):
    #         if train_data[i, j, 30] == 1:
    #             count += 1
    # print('count:', count)

    #滑动窗口技术
    size = np.size(train_data, axis=0)*(700 - window_size +1)
    x_train = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_train = np.zeros((size, 9))
    #训练集
    flag = 0
    for i in range(np.size(train_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = train_data[i, j+k, :22]
                feature[k, ] = train_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    y_train[flag, ] = train_data[i, j+k, 22:31]
            x_train[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1
    #测试集
    size = np.size(test_data, axis=0)*(700 - window_size +1)
    x_test = np.zeros((size, window_size, 22+22))#22氨基酸独热编码+22PSSM谱编码
    seq = np.zeros((window_size, 22))
    feature = np.zeros((window_size, 22))
    y_test = np.zeros((size, 9))

    flag = 0
    for i in range(np.size(test_data, axis=0)):
        for j in range(700 - window_size):
            for k in range(window_size):
                seq[k, ] = test_data[i, j+k, :22]
                feature[k, ] = test_data[i, j+k, 35:57]
                if k == int(window_size/2):
                    y_test[flag, ] = test_data[i, j+k, 22:31]
            x_test[flag, ] = np.concatenate((seq, feature), axis = 1)
            flag += 1

    # count = 0
    # size = np.size(y_test, axis=0)
    # for i in range(size):
    #     if y_test[i, 8] == 1:
    #         count += 1
    # print('count:', count)


    print("x_train, y_train, x_test, y_test data shape:")
    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))
    print('Data processing completed')

    # print(x_valid[0, 8, 0:43])
    # print(y_valid[0, 0:9])


    return x_train, y_train, x_test, y_test

def get_dssp_data():#读取数据
    print('reading data from cullpdb_chains6626_x_train_aminoacid.npy and  cullpdb_chains6626_y_trrain.npy ...')
    x_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_x_train_aminoacid.npy')#(3185724, 7, 28)
    y_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_y_trrain.npy')#(3185724, 8)

    np.set_printoptions(threshold=1000000)
    # 个samples，每个sample有700个氨基酸
    x_train_data.shape = (-1, 7, 28)
    y_train_data.shape = (-1, 8)
    # with open("C:\\bishe\data\dssp\data", 'a', encoding='gb18030') as f:
    #     f.write(str(input_datas[0:2, 0:700, :]))

    x_train = x_train_data[ : 2480000].astype(np.float32)#[0,5600）训练
    y_train = y_train_data[ : 2480000].astype(np.float32)
    x_test = x_train_data[2480000 : 3100000].astype(np.float32)#[5605,5877）=272
    y_test = y_train_data[2480000 : 3100000].astype(np.float32)#[5877,6133]验证 = 256

    return x_train, y_train, x_test, y_test

def get_seq_data():
    print('reading data from cullpdb_chains6626_x_train_aminoacid.npy and  cullpdb_chains6626_y_trrain.npy ...')
    x_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_x_train_aminoacid.npy')#(3185724, 7, 28)
    y_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_y_trrain.npy')#(3185724, 8)

    np.set_printoptions(threshold=1000000)
    # 个samples，每个sample有700个氨基酸
    x_train_data.shape = (-1, 7, 28)
    # x_train_data = x_train_data[:, :, 0:22]
    y_train_data.shape = (-1, 8)

    x_train = x_train_data[ : 2480000].astype(np.float32)#[0,5600）训练
    y_train = y_train_data[ : 2480000].astype(np.float32)
    x_test = x_train_data[2480000 : 3100000].astype(np.float32)#[5605,5877）=272
    y_test = y_train_data[2480000 : 3100000].astype(np.float32)#[5877,6133]验证 = 256

    return x_train, y_train, x_test, y_test

def get_data():
    print('reading data from cullpdb_chains6626_x_train_aminoacid.npy and  cullpdb_chains6626_y_trrain.npy ...')
    x_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_x_trrain.npy')#(3185724, 7, 28)
    y_train_data = np.load('C:\\bishe\data\\available\cullpdb_chains6626_y_trrain.npy')#(3185724, 8)

    np.set_printoptions(threshold=1000000)
    x_train_data.shape = (-1, 7, 28)
    y_train_data.shape = (-1, 8)

    x_train = x_train_data[ : 2480000].astype(np.float32)#[0,5600）训练
    y_train = y_train_data[ : 2480000].astype(np.float32)
    x_test = x_train_data[2480000 : ].astype(np.float32)#[5605,5877）=272
    y_test = y_train_data[2480000 : ].astype(np.float32)#[5877,6133]验证 = 256

    return x_train, y_train, x_test, y_test

def filter_Noseq():
    test_file = 'C:\\bishe\data\\available\cb513+profile_split1.npy.gz'
    print('reading data...')
    data = np.load(test_file)

    np.set_printoptions(threshold=1000000)
    # 6133个samples，每个sample有700个氨基酸
    data.shape = (-1, 700, 57)
    data = data[ : ].astype(np.float32)

    #测试集
    count = 0
    datas = []
    for i in range(np.size(data, axis=0)):
        temp = []
        for j in range(700):
            if data[i, j, 30] != 1:
                temp.append(data[i, j, ])
                # count += 1

        temp = np.array(temp)
        print(np.shape(temp))
        datas.append(temp)
    datas = np.array(datas)
    print(np.size(datas))

# get_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
# get_cnn_data(FILE_NAME, TRAIN_SET, TEST_SET, VALID_SET)
# get_cb513_data(9)
# filter_Noseq()
get_cb513_3test_data(9)
