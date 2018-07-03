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

window_size = 13
#9：（3096479，9，26）
#17: (3045241, 17, 26)
#13: (3071002, 13,26)

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
def get_seq_onehot(seq):
    feature = []
    strs = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(len(strs)):
        if seq == strs[i]:
            feature.append(1)
        else:
            feature.append(0)
    return feature

#标签的独热编码
def get_label_onehot(seq):
    label = []
    strs = '    '
    # labels = []
    for i in range(len(strs)):
        if seq == strs[i]:
            label.append(1)
        else:
            label.append(0)
    return label

#PDBID = 2A0B
def get_PDBid_feature(PDBid):
    count = 0
    seqs = []
    labels = []
    aminoacids = 'ACDEFGHIKLMNPQRSTVWY'

    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % PDBid)
    for i in cursor:
        temp = i[1]
        label = i[2]
        for j in aminoacids:
            if j == temp:
                count += 1
                seqs.append(temp)
                labels.append(label)
    # print(PDBid,'的长度为：',count)
    # print(seqs)
    #用矩阵存储信息,20个氨基酸独热+8个feature
    pdb_xtrain = np.zeros((count, 20+6))
    pdb_ytrain = np.zeros((count - window_size +1, 8))

    count1 = 0
    #每个循环为矩阵添加一行
    for temp in seqs:
        feature = get_seq_onehot(temp)
        for i in aminoacid[temp]:#6个属性
            feature.append(i)
        pdb_xtrain[count1, ] = feature
        count1 += 1

    # print(count,count1,count-(int(window_size/2)))
    count2 = 0
    for temp in range((int(window_size/2)), count-(int(window_size/2))):
        # if count2 >= (int(window_size/2)) & count2 < (count-(int(window_size/2))):
        label = get_label_onehot(labels[temp])
        pdb_ytrain[count2, ] = label
        count2 += 1
    # print(count,count1,count2)
    return pdb_xtrain, pdb_ytrain

#滑动窗口默认按照每7个滑动,形成shape = (-1, 7, 28)
def slide_window(pdb_xtrain):
    size = np.size(pdb_xtrain, axis=0)
    xtrain = np.zeros((size-window_size+1, window_size, 26))
    x = np.zeros((window_size, 26))
    for i in range(size - window_size):
        for j in range(window_size):
            x[j, ] = pdb_xtrain[i+j, ]
        xtrain[i, ] = x
    return xtrain

def get_PDBid():#从数据库中获取PDB
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

    return line


if __name__ == '__main__':#x_train 20+6,start time:15:20,end time:
    x_train = np.zeros((3148131, window_size, 26))#3148131去掉所有非20种,再减去每条链前后的windowsize +1(共6626条链)
    y_train = np.zeros((3148131, 8))
    read_excel()

    #获取所有PDBid
    PDB_id_lines = get_PDBid()
    count = 0
    count1 = 0
    count2 = 0
    aminoacids = 'ACDEFGHIKLMNPQRSTVWY'

    for pdb in PDB_id_lines:
    # for pdb in range(1):
        # pdb = '2A0B'
        if pdb == '4V4M':
            continue
        t = 0
        cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id ='%s' " % pdb)
        for i in cursor:
            temp = i[1]
            for j in aminoacids:
                if j == temp:
                    t += 1
        if t < 15:
            continue
        print('reading:',pdb, count ,'   ',count1,'/',3148131)
        pdb_xtrain, pdb_ytrain = get_PDBid_feature(pdb)

        pdb_xtrain = slide_window(pdb_xtrain)
        # print('re:', np.shape(pdb_xtrain),np.shape(pdb_ytrain))
        for k in range(np.size(pdb_xtrain, axis=0)):
            x_train[count1, ] = pdb_xtrain[k]
            count1 += 1

        for i in pdb_ytrain:
            y_train[count2, ] = i
            count2 += 1

        count += 1
    conn.close()
    #将x_train,y_train 数组保存到文件中
    np.save('C:\\bishe\data\\available\cullpdb_chains6626_x_train_13_26_aminoacid.npy', x_train)
    np.save('C:\\bishe\data\\available\cullpdb_chains6626_y_train_13_26_aminoacid.npy', y_train)
    print(np.shape(x_train),np.shape(y_train))
    print('x_train y_train successful!')
    # print(x_train[:1, :,:])
    # print(y_train[0:20, :])

# read_excel()
# get_PDBid_feature()