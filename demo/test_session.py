#-*- coding:utf-8 -*-
import tensorflow as tf

#创建两个矩阵

matrix1 = tf.constant([[5,5]]) #一行两列
matrix2 = tf.constant([[2],[2]])#两行一列

#矩阵乘法
product = tf.matmul(matrix1,matrix2)

#方式一：
sess = tf.Session()
result = sess.run(product)
sess.run(product)
sess.close()

#方式二
# with tf.Session() as sess:
# 	#在with块中不用关闭sess 默认使用了close 方法。
#     result2 = sess.run(product)
#     print(result2)
