#-*- coding:utf-8 -*-
import tensorflow  as tf

"""
Fetch:
	为了取回操作的输出内容,
	可以在使用 Session 对象的 run() 调用 执行图时,
	传入一些 tensor, 这些 tensor 会帮助你取回结果.
"""

const1 = tf.constant(3.0)
const2 = tf.constant(2.0)
const3 = tf.constant(5.0)

intermed = tf.add(const2,const3)
mul = tf.mul(const1,intermed)

with tf.Session() as sess:
	result = sess.run([mul,intermed])
	print result

"""
Feed:
	上面计算图中引入了 tensor, 以常量或变量的形式存储. 
	TensorFlow 还提供了 feed 机制, 
	该机制可以临时替代图中的任意操作中的 tensor 
	可以对图中任何操作提交补丁, 直接插入一个 tensor

"""
# #教程中的tf.types.float32是错误的
feed1 = tf.placeholder(tf.float32)
feed2 = tf.placeholder(tf.float32)
output = tf.mul(feed1,feed2)

with tf.Session() as sess:
	print sess.run([output],feed_dict={feed1:[7.],feed2:[2.]})



