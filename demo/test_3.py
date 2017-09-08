#-*- coding:utf-8 -*-
import tensorflow as tf

#定义变量 给定初始值和名字
state = tf.Variable(0,name='state_name')

print state.name #state_name:0

#定义一个常量
const_1 = tf.constant(1)

#变量+常量 = 变量
new_value = tf.add(state,const_1) #定义加法步骤 (注: 此步并没有直接计算)
# 将 State 更新成 new_value 
update = tf.assign(state,new_value)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)#激活init
	for i in range(10):
		sess.run(update)
		print sess.run(state)
