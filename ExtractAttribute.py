# !/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import xlrd

def read_excel():#从excel中对文件,将对应氨基酸及其属性,以键值对的方式写入字典中
    # 打开文件D:\bishe\Aminoacid.xlsx
    fileName = input('氨基酸属性表路径：')
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
    print(aminoacid)

def read_amino_acid_seq(fileDir): #从文件中读氨基酸序列
    # seq_len = int(seq_len)
    # i = 0
    for (root, dirs, files) in os.walk(fileDir):  #列出windows目录下的所有文件和文件名
        for filename in files:
            print('正在读取'+filename+'...')
            file = fileDir+'\\'+filename
            with open(file, 'r') as f:
                lines = f.readlines()
                line = lines[1]#文件中的氨基酸序列
                get_attributions(line,filename)
    print('读取完成！')

def get_attributions(line, filename): #获取氨基酸序列的属性
    with open(path+ '\\'+filename, 'a', encoding='gb18030') as f:
        lines_len = line.__len__() - 1
        j = 0
        for i in range(int(lines_len / seq_len)):
            if (j + seq_len < lines_len):
                # strs = 'ARN'
                strs = line[j:j + seq_len]
                for k in range(6):
                    for x in strs:
                        f.write(str(aminoacid[x][k]) + ' ')
                    f.write('|')
                f.write('\n')
                j += 3
                # print(j)

def write_file(attributes, filename):#将特征值写入文件
    attributes = str(attributes)
    #6个数据通道名字，以键值对的方式
    # fileName = ['Side chain class', 'Side chain polarity[136]', 'Side chain charge (pH 7.4)[136]','Hydropathy index[137]','MW (weight)', 'Occurrence in  proteins (%)[139]']
    # fileName = 'seq' + seq
    #a 以追加模式打开 (从 EOF 开始, 必要时创建新文件)
    # path = path + '\\'+fileName
    with open(path+ '\\'+filename, 'a', encoding='gb18030') as f:
        f.write(attributes+'|')

def mkdir(path):#判断是对应长度的氨基酸序列文件夹是否存在
    #去除首位空格
    path = path.strip()
    #去除尾部\符号
    path = path.rstrip('\\')
    print(path)
    #判断路径是否存在
    #存在     True
    #不存在    False
    isExists = os.path.exists(path)
    # print(isExists)
    #判断结构
    if not isExists:
        #如果不存在创建目录
        #创建目录操作函数
        os.makedirs(path)
        return path
    else:
        return path

if __name__ == "__main__":#主函数
    # 定义使用字典（键值对），存放氨基酸属性
    aminoacid = {}
    seq_len = int(input('氨基酸长度：'))
    seq = str(seq_len)
    # seq_len = 3
    read_excel()
    # fileDir = "D:\\bishe\\allfasta"
    fileDir = input('氨基酸序列所在文件夹路径：')
    #对应长度的氨基酸序列文件夹
    mkpath = input('输入提取特征值储存路径：')+seq
    # mkpath = 'D:\\bishe\\attribution\\seq' + seq
    path = mkdir(mkpath)
    read_amino_acid_seq(fileDir)