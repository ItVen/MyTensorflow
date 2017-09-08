#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

#create some　date
x_data = np.random.rand(100).astype(np.float32) 
y_data = x_data*0.5+0.7

#创建tensorflow结构开始
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#一维结构，随机数列的范围　－１～１
biases = tf.Variable(tf.zeros([1]))#初始在为０
#通过学习，Ｗ，Ｂ将不段的提升到我们接近我们的y_data

y = Weights*x_data+biases


#计算 y 和 y_data 的误差:
loss = tf.reduce_mean(tf.square(y-y_data))

#建立优化器减少误差 提升参数的准确度
optimizer= tf.train.GradientDescentOptimizer(0.1)#误差传递方法是梯度下降法: Gradient Descent 
train = optimizer.minimize(loss) #使用 optimizer 来进行参数的更新

#初始化变量
init = tf.global_variables_initializer()

#创建tensorflow结构结束

#定义会话
sess = tf.Session()
sess.run(init) #激活神经网络　很重要

#训练
for step in range(501):
	#用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
	sess.run(train)
	if step%20 == 0:
		print (step,sess.run(Weights),sess.run(biases))

