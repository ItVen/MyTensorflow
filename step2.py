# -*- coding:utf-8 -*-
'''
链接：http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
下载下来的数据集被分成两部分：60000行的训练数据集（mnist.train）
和10000行的测试数据集（mnist.test）。这样的切分很重要，在机器
学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评
估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）
'''
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)