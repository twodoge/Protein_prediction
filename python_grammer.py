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

#添加神经层的函数，参数（输入值、输入的形状、输出的形状、激励函数）
def add_layer(inputs, in_size, out_size, activation_function = None):
    #tf.random_normal()参数为shape，还可以指定均值和标准差
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def sample_2():#构建一个神经网络
    #构建数据集
    #np.linspace()在-1和1之间等差生成300个数字
    #noise是正态分布的噪音，前两个参数是正态分布的参数，然后是size,newaxis:np.newaxis的功能是插入新维度
    x_data = np.linspace(-1, 1 , 300, dtype=np.float32)[: , np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 +noise

    #利用占位符定义我们所需的神经网络输入。
    #第二个参数为shape，：None代表行数不定，1是列数。
    #这里的行数就是样本数，列数就是每个样本的特征数。
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    #输入层1个神经元（因为只有一个特征），隐藏层是10个，输出层是1个。
    #调用函数定义隐藏层和输出层，输入size是上一层的神经元个数（全连接），输出size是本层个数。
    l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
    prediction = add_layer(l1, 10, 1, activation_function = None)

    #计算预测值predition和真实值的误差， 对二者差的平方求和再取平均作为损失函数
    #reduction_indeics表示最后数据的压缩维度，好像一般不用这个参数（即降到0维，一个标量）
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #初始化变量、激活、执行运算
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            #training
            sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
            if i % 50 == 0:
                print (sess.run(loss, feed_dict={xs:x_data,ys:y_data}))

def about_tensor():
    '''
    规模最小的张量是0阶张量，即标量，也就是一个数。

    当我们把一些数有序的排列起来，就形成了1阶张量，也就是一个向量

    如果我们继续把一组向量有序的排列起来，就形成了2阶张量，也就是一个矩阵

    把矩阵摞起来，就是3阶张量，我们可以称为一个立方体，具有3个颜色通道的彩色图片就是一个这样的立方体
    '''
    #沿着某轴是什么意思
    a = np.array([[1,2], [3,4]])
    sum0 = np.sum(a, axis=0)
    sum1 = np.sum(a, axis=1)

    print(sum0)
    print(sum1)

if __name__ == "__main__":#主函数
    about_tensor()