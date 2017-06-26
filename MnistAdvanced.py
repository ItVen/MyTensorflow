# -*- coding:utf-8 -*-
'''
MNIST 进阶
TensorFlow擅长的任务之一就是实现以及训练深度神经网络。
本次任务 
	构建TensorFlow模型的基本步骤，并通过该步骤为MNIST构建
	一个深度学习的神经网络

mnist 是一个轻量级的类
	以Numpy 数组的形式存储 训练 校验 测试 数据集 

TensorFlow 的一般流程
	创建一个图 在session中启动

InteractiveSession类  （交互式会话）
	可以在运行图的时间，插入计算图


'''
import tensorflow as tf
# 加载MNIST 数据集
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 构建Softmax回归模型
# 创建一个拥有一个线性层的回归模型

# 开始创建计算图

x = tf.placeholder("float",shape =[None,784])
y_ = tf.placeholder("float",shape =[None,10])

# 设置模型参数
W = tf.Variable(tf.zeros([784,10])) # 784 个特征 10个输出值
b = tf.Variable(tf.zeros([10])) #10维向量 10个分类

# 初始化参数，并在session中启动

sess.run(tf.initialize_all_variables())

# 实现回归模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 实现交叉炳 ----为训练过程指定最小化误差用的损失函数 	
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练模型

# 使用自动微分法找到对于各个变量的损失的梯度值  --最速下降法让交叉熵下降，步长为0.01.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 反复训练1000次

for i in range(1000):
	# 加载50个训练样本
	batch = mnist.train.next_batch(50)
	# 并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
	train_step.run(feed_dict ={x:batch[0],y_:batch[1]})

# 评估模型
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值

# 比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))	# 返回一组布尔值

# 将布尔值转换为浮点数来代表对、错，然后取平均值

accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

# 计算测试的准确值
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# ==>0.9092 正确率只有91% 不理想,使用稍微复杂的模型,提高准确率
# 卷积神经网络