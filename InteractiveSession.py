# -*- coding:utf-8 -*-

# 交互式使用
'''
为了便于使用诸如 IPython 之类的 Python 交互环境, 
可以使用 InteractiveSession 代替 Session 类, 
使用 Tensor.eval() 和 Operation.run() 方法 代替 Session.run() . 
这样可以避免使用一个变量来持有会话.
'''

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

x.initializer.run()

sub = tf.sub(x,a)

print sub.eval()