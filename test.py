#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# print('hello')
# f = open('python_io.txt', 'r')
# print(f.read())
# f.close()
import os

def read_amino_acid_seq(fileDir, seq_len): #从文件中读氨基酸序列
    for (root, dirs, files) in os.walk(fileDir):  #列出windows目录下的所有文件和文件名
        for filename in files:
            file = fileDir+'\\'+filename
            with open(file, 'r') as f:
                lines = f.readlines()
                line = lines[1]
                lines_len = lines[1].__len__()-1
                for i in range(lines_len):
                    if(i+2 < lines_len):
                        print(line[i:i+seq_len])
        for dirc in dirs:
            print(os.path.join(root,dirc))

fileDir = "D:\\bishe\\allfasta"
seq_len = 3
# read_amino_acid_seq(fileDir, seq_len)

#键值对
d = {
    'Michael': 95,
    'Bob': 75,
    'Tracy': 85
}
print('d[\'Michael\'] =', d['Michael'])
print('d[\'Bob\'] =', d['Bob'])
print('d[\'Tracy\'] =', d['Tracy'])
print('d.get(\'Thomas\', -1) =', d.get('Thomas', -1))

str = 'abc'
for x in str:
    print(x)