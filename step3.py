#-*- coding:utf-8 -*-
'''
MNIST 数据集 
	每一张图片是[28,28]个像素的数组，展开成一个向量 长度是784
	mnist.train.images 是[60000,784]的张量 
		一维----图片索引，二维---- 该索引下的图片像素点
Softmax 回归
	特定的数字类证据 evidence
	对图片像素进行加权求和  + 偏置量
		权值负： 不属于该类
		权值正： 属于该类
	y = softmax(evidence) 返回概率y
	y = softmax(Wx+b)

回归模型

训练模型
评估模型

'''
# 回归模型
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float',[None,784])

# 变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 实现模型 matmul==矩阵相乘
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 训练模型
# 交叉炳
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# GradientDescentOptimizer 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.initialize_all_variables()

# Session 里启动模型

sess = tf.Session()
sess.run(init)

# 训练模型 循环训练1000次

for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict ={x: batch_xs,y_:batch_ys})

# 评估模型

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

# 将布尔值转化成浮点数后取平均值

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})

