#-*- coding:utf-8 -*-
import tensorflow as tf


# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input_1 = tf.placeholder(tf.float32)#可以指定类型和机构
input_2 = tf.placeholder(tf.float32)

# multiply 是将input1和input2 做乘法运算，并输出为 output 
output = tf.multiply(input_1,input_2)

with tf.Session() as sess:
	print sess.run(output,feed_dict={input_1:4.0,input_2:7.0})#传入值