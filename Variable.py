#-*- coding:utf-8 -*-
'''
演示如何使用变量实现一个简 单的计数器
'''

import tensorflow as tf

state = tf.Variable(0,name="counter")

one = tf.constant(1)

new_value = tf.add(state,one)

update = tf.assign(state,new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(state)
	# 运行 op, 更新 'state', 并打印 'state'
	for _ in range(10):
		sess.run(update)
		print sess.run(state)

'''
assign() 操作是图所描绘的表达式的一部分,
正如 add() 操作一样. 所以在调用 run() 执行表达式 之前,
它并不会真正执行赋值操作.
'''
