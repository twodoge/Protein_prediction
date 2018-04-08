#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import sys
import tensorflow as tf

#numpy shape
def shape():#返回矩阵的形状，即行列数
    x = np.array([[1, 2, 5], [2, 3, 5], [3, 4, 5], [2, 3, 6]])
    # 输出数组的行和列数
    print (x.shape)  # (4, 3)
    # 只输出行数
    print (x.shape[0]) # 4
    # 只输出列数
    print (x.shape[1]) # 3

def sys_argv():#Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的
    a = sys.argv[0]
    print(a)
    # if a = sys.argv[1] then 再从控制台窗口运行，这次我们加上一个参数，输入：test.py what
    # 输出 what

def tensorflow_grammer():
    # 定义一个tensorflow的变量：
    state = tf.Variable(0, name='counter')
    # 定义常量
    one = tf.constant(2)
    # 定义加法步骤 (注: 此步并没有直接计算)
    new_value = tf.add(state, one)
    # 将 State 更新成 new_value
    update = tf.assign(state, new_value)
    # 变量Variable需要初始化并激活，并且打印的话只能通过sess.run()：
    init = tf.global_variables_initializer()
    # 使用 Session 计算
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

if __name__ == "__main__":#主函数
    tensorflow_grammer()