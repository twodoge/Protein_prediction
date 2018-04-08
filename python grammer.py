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

def sample_1():# 例子1，拟合y_data的函数，权重和偏置分别趋近0.1和0.3
    #np.random.rand(100) 生成100个0-1之间的随机数，构成一维数组
    #mp.random.rand(2,3) 生成2行3列的二维数组
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3

    '''权重偏置这些不断更新的值用tf变量存储，
    tf.random_uniform()的参数意义为：（shape,min,max）
    偏置初始化为0
    '''
    weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    biases = tf.Variable(tf.zeros([1]))
    # print(weights)
    # print(biases)

    y = weights * x_data +biases

    #损失函数 tf.reduce_mean()是取平均值
    loss = tf.reduce_mean(tf.square(y - y_data))

    #用梯度优化方法最小化损失函数 learning_rate
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    #tf变量是需要初始化的，而且后边计算时还需要sess.run()一下
    init = tf.global_variables_initializer()

    #Session进行计算
    with tf.Session() as sess:
        sess.run(init)
        for step in range(200):
            sess.run(train)
            if step in range(200):
                if step%20 == 0:
                    print (step, sess.run(weights), sess.run(biases))

if __name__ == "__main__":#主函数
    sample_1()