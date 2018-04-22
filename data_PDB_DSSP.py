#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import sqlite3 as sql

FILENAME = 'C:\\bishe\data\\available\cullpdb_pc20_res2.0_R0.25_d180322_chains6626.fasta'

#创建数据库，存储蛋白质的氨基酸序列的二姐结构和属性
def create_protein_database():
    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    # if c.execute(" SELECT COUNT(*) FROM sqlite_master where type='table' and name='PDB_id' " ) == 0:
    c.execute('CREATE TABLE PDB_id (ID INTEGER PRIMARY KEY autoincrement, PDB_id char(8) NOT NULL)')
    c.execute('CREATE  TABLE  PDB_DSSP ('
              # 'ID INTEGER PRIMARY KEY autoincrement , '
              'PDB_id char(8) NOT NULL,'
              'amino_acid_seq char(2) NOT NULL ,'
              'structs char(2) NOT NULL , '
              'TOO real NOT NULL, '
              'KAPPA REAL NOT NULL , '
              'ALPHA REAL NOT NULL , '
              'PHI REAL NOT NULL , '
              'PSI REAL NOT NULL , '
              'X_CA REAL NOT NULL ,'
              'Y_CA REAL NOT NULL ,'
              'Z_CA REAL NOT NULL'
                  ')')
    conn.commit()
    conn.close()
    print ("create table successful")

#将序列的信息写入PDB_DSSP表格
def write_in_PDB_DSSP_table(data, c):
    # conn = sql.connect('protein_database.db')
    # c = conn.cursor()
#data = {'PDB_id':PDB_id, 'seq':seq, 'sturct':sturct, 'TCO':TCO, 'KAPPA':KAPPA, 'ALPHA':ALPHA, 'PHI':PHI, 'PSI':PSI, 'X_CA':X_CA, 'Y_CA':Y_CA, 'Z_CA':Z_CA}
    c.execute(" INSERT INTO PDB_DSSP VALUES ('%s', '%s', '%s', '%f', '%f', '%f','%f','%f','%f','%f','%f')"
              % (data['PDB_id'], data['seq'], data['sturct'], data['TCO'], data['KAPPA'], data['ALPHA'], data['PHI'], data['PSI'], data['X_CA'], data['Y_CA'], data['Z_CA']))
    # conn.commit()
    # conn.close()
    print('DSSP store finish!')

#将PDBID写入PDB_id表
def write_in_PDBid_table(seq):
    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    c.execute(" INSERT INTO PDB_id (PDB_id) VALUES ('%s')" % (seq))
    # cursor = c.execute(' SELECT PDB_id FROM PDB_id')
    # for row in cursor:
    #     print(row[0])
    conn.commit()
    conn.close()

def get_PDB_id_fasta(filename):
    print('get PDBid from ', filename)

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('>'):
                seq = line[1:6]
                write_in_PDBid_table(seq)# 1:5为PDBid,5:6为A\B链
            line = f.readline()
    print('read success!')

def select_PDBid():
    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    cursor = c.execute("SELECT * FROM PDB_DSSP WHERE PDB_id= '%s' " % '3A0Y')
    count = 0
    for row in cursor:
        print(row)
        count += 1
    print(count)
    conn.commit()
    conn.close()

def delete_PDBid():
    seq = '4V4M'
    conn = sql.connect('protein_database.db')
    c = conn.cursor()
    cursor = c.execute("DELETE FROM PDB_DSSP WHERE PDB_id = '%s'"% seq)
    # for row in cursor:
    #     print(row)
    print('succu')
    conn.commit()
    conn.close()

# create_protein_database()
# get_PDB_id_fasta(FILENAME)
select_PDBid()
# delete_PDBid()
# main()
