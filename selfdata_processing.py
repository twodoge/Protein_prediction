#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import sqlite3 as sql
import data_PDB_DSSP

#1 H = α-helix  HBEGITS
#2 B = residue in isolated β-bridge
#3 E = extended strand, participates in β ladder
#4 G = 3-helix (310 helix)
#5 I = 5 helix (π-helix)
#6 T = hydrogen bonded turn
#7 S = bend
#8 '' 无
'''
氨基酸残基的顺序是ACDEF GHIKL MNPQR STVWXY+Noseq = 22
'''

conn = sql.connect('protein_database.db')
c = conn.cursor()

def get_PDBid():#从数据库中获取PDB
    # conn = sql.connect('protein_database.db')
    # c = conn.cursor()
    cursor = c.execute('SELECT PDB_id FROM PDB_id')
    count = 0
    line = []
    count1 = 0
    same = 'abcd' #用于判断相同的PDBid
    for row in cursor:
        count1 += 1
        row = row[0]
        if row[0:4] != same:
            line.append(row[0:4])
            count += 1
        same = row[0:4]
    print('diferent id:',count)
    print(count1)
    # print('pdb:',count) #6626
    # conn.close()
    # for i in range(0, count):
    #     PDB_id = line[i]
        # print('READING:',PDB_id, '   ', i,'/6626')

    return line
    # conn.commit()
    # conn.close()

def get_PDB_DSSP():#从数据库中获取PDB 3185724
    # conn = sql.connect('protein_database.db')
    # c = conn.cursor()
    cursor = c.execute('SELECT PDB_id FROM PDB_DSSP')
    count = 0
    for row in cursor:
        count += 1
    # print(count)
    # conn.commit()
    # conn.close()

#氨基酸的独热编码
def get_seq_onehot(seq, feature):
    strs = 'ACDEFGHIKLMNPQRSTVWXY '
    seqs = []
    for i in range(len(strs)):
        if seq == strs[i]:
            feature.append(1)
        else:
            feature.append(0)
    return feature
#标签的独热编码
def get_label_onehot(seq, label):
    strs = 'HBEGITS '
    # labels = []
    for i in range(len(strs)):
        if seq == strs[i]:
            label.append(1)
        else:
            label.append(0)
    return label


#PDBID = 2A0B
def get_PDBid_feature(PDBid):
    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % PDBid)
    count = 0
    for i in cursor:
        count += 1
    # print(PDBid,'的长度为：',count)
    #用矩阵存储信息,22个氨基酸独热+8个feature， count+3+3：前后补3个Noseq
    pdb_xtrain = np.zeros((count + 6, 22+8))
    #虽然前后补上Noseq，但是还是分8类，因为并没有对前后不上的进行预测
    pdb_ytrain = np.zeros((count, 8))

    count1 = 0
    count2 = 0
    #前补3个Noseq
    feature = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,1, 0,0,0,0,0,0,0,0]
    for i in range(3):
        pdb_xtrain[count1, ] = feature
        count1 += 1
    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % PDBid)
    #每个循环为矩阵添加一行
    for row in cursor:
        # print(row)
        feature = []
        label = []

        #序列和属性
        get_seq_onehot(row[1], feature)
        for i in range(3,11):
            feature.append(row[i])
        # print(feature)
        pdb_xtrain[count1, ] = feature

        #标签
        get_label_onehot(row[2], label)
        pdb_ytrain[count2, ] = label
        # print(label)

        count1 += 1
        count2 += 1
        # print(row)

    # 后补3个Noseq
    feature = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,1, 0,0,0,0,0,0,0,0]
    for i in range(3):
        pdb_xtrain[count1,] = feature
        count1 += 1

    # conn.close()
    # pdb_xtrain = (-1,30),pdb_ytrain =(-1 ,8)
    # print(pdb_xtrain)
    # print(pdb_ytrain)
    return pdb_xtrain, pdb_ytrain

#滑动窗口默认按照每7个滑动,形成shape = (-1, 7, 30)
def slide_window(pdb_xtrain):
    size = np.size(pdb_xtrain, axis=0)
    xtrain = np.zeros((size-6, 7, 30))
    x = np.zeros((7, 30))
    for i in range(size - 6):
        for j in range(7):
            x[j, ] = pdb_xtrain[i+j, ]
        xtrain[i, ] = x
    return xtrain

if __name__ == '__main__':#x_train = (3185724, 7, 30),y_train = (3185724, 7, 8),start time:15:20,end time:
    x_train = np.zeros((3185724, 7, 30))
    y_train = np.zeros((3185724, 8))

    #获取所有PDBid
    PDB_id_lines = get_PDBid()
    count = 0
    count1 = 0
    count2 = 0
    for pdb in PDB_id_lines:
    # for i in range(1):
        print('reading:',pdb, count,'   ',count1, count2,'/ 3185724')
        pdb_xtrain, pdb_ytrain = get_PDBid_feature(pdb)
        # print(np.shape(pdb_ytrain))

        pdb_xtrain = slide_window(pdb_xtrain)
        if np.size(pdb_xtrain, axis=0) != np.size(pdb_ytrain,axis=0):
            print('erroe')
            break
        # print('re:', pdb_xtrain[0])
        for k in range(np.size(pdb_xtrain, axis=0)):
            x_train[count1, ] = pdb_xtrain[k]
            count1 += 1

        for j in range(np.size(pdb_ytrain, axis=0)):
            # print('re:',pdb_ytrain[j])
            y_train[count2, ] = pdb_ytrain[j]
            count2 += 1

        count += 1

    conn.close()

    #将x_train,y_train 数组保存到文件中
    np.save('cullpdb_chains6626_x_trrain.npy', x_train)
    np.save('cullpdb_chains6626_y_trrain.npy', y_train)

    print(np.shape(x_train))
    print(np.shape(y_train))
    print('x_train,y_train successful!')
    # print(x_train[:1, :,:])
    # print(y_train[0:20, :])

# get_PDBid_feature()
# get_PDBid()
# get_PDB_DSSP()