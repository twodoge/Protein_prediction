#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import os
import xlrd 
# import selfdata_processing
import sqlite3 as sql

aminoacid = {}
conn = sql.connect('protein_database.db')
c = conn.cursor()

def read_excel():#从excel中对文件,将对应氨基酸及其属性,以键值对的方式写入字典中
        # 打开文件D:\bishe\Aminoacid.xlsx
    fileName = 'C:\\bishe\data\Aminoacid.xlsx'
    workbook = xlrd.open_workbook(fileName)
    sheet = workbook.sheet_by_name('Sheet1')
    # print(sheet.name, sheet.nrows, sheet.ncols)

    # 获取20个氨基酸
    cols = sheet.col_values(2)
    cols.pop(0)
    cols.pop(-1)
    cols.pop(-1)

    # print(cols)
    i = 1
    for x in cols:
        rows = sheet.row_values(i)
        rows.pop(0)
        rows.pop(0)
        rows.pop(0)
        rows.pop(0)
        rows.pop(1)
        rows.pop(2)
        rows.pop(4)
        rows.pop(4)
        # print(rows)
        aminoacid.setdefault(x, rows)
        i = i+1
    # print(aminoacid)
    a = 'A'
    # print(aminoacid[a][1])

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

#PDBID = 2A0B
def get_PDBid_feature(PDBid):
    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % PDBid)
    count = 0
    aminoacids = 'ACDEFGHIKLMNPQRSTVWY'
    for i in cursor:
        count += 1
    # print(PDBid,'的长度为：',count)
    #用矩阵存储信息,22个氨基酸独热+8个feature， count+3+3：前后补3个Noseq
    pdb_xtrain = np.zeros((count + 6, 22+6))
    #虽然前后补上Noseq，但是还是分8类，因为并没有对前后不上的进行预测

    count1 = 0
    #前补3个Noseq
    feature = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,1, 0,0,0,0,0,0]
    for i in range(3):
        pdb_xtrain[count1, ] = feature
        count1 += 1
    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % PDBid)
    #每个循环为矩阵添加一行
    for row in cursor:
        # print(row)
        feature = []
        temp = row[1]

        #序列和属性
        get_seq_onehot(row[1], feature)
        #XX表示任意一种氨基酸或者不确定种类，我们把X当成A处理
        for i in range(20):
            if temp != aminoacids[i]:
                temp = 'A'
        for i in aminoacid[temp]:
            feature.append(i)
        # print(feature)
        pdb_xtrain[count1, ] = feature
        count1 += 1

    # 后补3个Noseq
    feature = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,1, 0,0,0,0,0,0]
    for i in range(3):
        pdb_xtrain[count1,] = feature
        count1 += 1

    # conn.close()
    # pdb_xtrain = (-1,30),pdb_ytrain =(-1 ,8)
    # print(pdb_xtrain)
    return pdb_xtrain

#滑动窗口默认按照每7个滑动,形成shape = (-1, 7, 28)
def slide_window(pdb_xtrain):
    size = np.size(pdb_xtrain, axis=0)
    xtrain = np.zeros((size-6, 7, 28))
    x = np.zeros((7, 28))
    for i in range(size - 6):
        for j in range(7):
            x[j, ] = pdb_xtrain[i+j, ]
        xtrain[i, ] = x
    return xtrain

def get_PDBid():#从数据库中获取PDB
    # conn = sql.connect('protein_database.db')
    # c = conn.cursor()
    cursor = c.execute('SELECT PDB_id FROM PDB_id')
    count = 0
    line = []
    same = 'abcd' #用于判断相同的PDBid
    for row in cursor:
        row = row[0]
        if row[0:4] != same:
            line.append(row[0:4])
            count += 1
        same = row[0:4]
    print('diferent id:',count)
    # print('pdb:',count) #6626
    # conn.close()
    # for i in range(0, count):
    #     PDB_id = line[i]
        # print('READING:',PDB_id, '   ', i,'/6626')

    return line

if __name__ == '__main__':#x_train = (3185724, 7, 28),y_train = (3185724, 7, 8),start time:15:20,end time:
    x_train = np.zeros((3185724, 7, 28))

    read_excel()

    #获取所有PDBid
    PDB_id_lines = get_PDBid()
    count = 0
    count1 = 0
    for pdb in PDB_id_lines:
    # for i in range(1):
        print('reading:',pdb, count ,'   ',count1,'/ 3185724')
        pdb_xtrain = get_PDBid_feature(pdb)

        pdb_xtrain = slide_window(pdb_xtrain)
        # print('re:', pdb_xtrain[0])
        for k in range(np.size(pdb_xtrain, axis=0)):
            x_train[count1, ] = pdb_xtrain[k]
            count1 += 1

        count += 1
    conn.close()
    #将x_train,y_train 数组保存到文件中
    np.save('cullpdb_chains6626_x_train_aminoacid.npy', x_train)

    print(np.shape(x_train))
    print('x_train successful!')
    # print(x_train[:1, :,:])
    # print(y_train[0:20, :])
