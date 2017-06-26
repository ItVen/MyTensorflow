#-*- coding:utf-8 -*-

"""
关于Tensorflow 必须知道的:
	使用 图( graph )表示计算任务
	在 绘画（session） 的上下文中执行认为
	使用tensor 表示数据
	通过变量 （variable） 维护数据
	使用 feed 和fetch 可以为任何操作赋值或者 获取数据
概述
	TensorFlow 是一个编程系统, 使用图来表示计算任务。
	图中的节点被称之为 op (operation 的缩写)。
	一个 op 获得 0 个或多个 Tensor，执行计算，产生 0 个或多个 Tensor。
	每个 Tensor 是一个类型化的多维数组。
	例如
		你可以将一小组图像集表示为一个四维浮点数数组，这四个维度分别是 [batch, height, width, channels]。

	一个 TensorFlow 图描述了计算的过程。
	为了进行计算， 图必须在 会话 里被启动。
	会话 将图的 op 分发到诸如 CPU 或 GPU 之类的 设备 上，同时提供执行 op 的方法。
	这些方法执行后，将产生的 tensor 返回。
	在 Python 语言中，返回的 tensor 是 numpy ndarray 对象；在 C 和 C++ 语言中，返回的 tensor 是 tensorflow::Tensor 实例。

计算图

	TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段。
	在构建阶段，op 的执行步骤 被描述成一个图。
	在执行阶段，使用会话执行执行图中的 op。
		例如：通常在构建阶段创建一个图来表示和训练神经网络，然后在执行阶段反复执行图中的训练 op。

构建图
	构建图的第一步，是创建源 op (source op)。
	源 op 不需要任何输入，例如 常量 (Constant)。
	源 op 的输出被传递给其它 op 做运算。

"""
# 构建图案例
import tensorflow as tf

# 创建一个常量 op，产生一个 1x2 矩阵。这个 op 被作为一个节点
# 加到默认图中。
#
# 构造器的返回值代表该常量 op 的返回值。
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op，产生一个 2x1 矩阵。
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op ， 把 'matrix1' 和 'matrix2' 作为输入。
# 返回值 'product' 代表矩阵乘法的结果。
product = tf.matmul(matrix1, matrix2)
"""
默认图现在有三个节点，两个 constant() op，和一个matmul() op。
为了真正进行矩阵相乘运算，并得到矩阵乘法的 结果，必须在会话里启动这个图。 

构造阶段完成后，才能启动图。
 启动图的第一步是创建一个 Session 对象，如果无任何创建参数，会话构造器将启动默认图。
"""

# 启动默认图
sess = tf.Session()
'''
调用 sess 的 'run()' 方法来执行矩阵乘法 op， 传入 'product' 作为该方法的参数。
上面提到, 'product' 代表了矩阵乘法 op 的输出， 传入它是向方法表明，我们希望取回矩阵乘法 op 的输出。
整个执行过程是自动化的， 会话负责传递 op 所需的全部输入。
op 通常是并发执行的。
函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行。

'''
# 返回值 'result' 是一个 numpy `ndarray` 对象。
result = sess.run(product)
print result

# 任务完成，关闭会话。
sess.close()
'''
Session 对象在使用完后需要关闭以释放资源。
除了显式调用 close 外，也可以使用 "with" 代码块 来自动完成关闭动作。
'''
with tf.Session() as sess:
  result = sess.run([product])
  print result

'''
如果机器上有超过一个可用的 GPU， 除第一个外的其它 GPU 默认是不参与计算的。
 为了让 TensorFlow 使用这些 GPU，你必须将 op 明确指派给它们执行。
 with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:
 	with tf.Session() as sess:
	  with tf.device("/gpu:1"):
	    matrix1 = tf.constant([[3., 3.]])
	    matrix2 = tf.constant([[2.],[2.]])
	    product = tf.matmul(matrix1, matrix2)

设备用字符串进行标识。目前支持的设备包括：
	"/cpu:0": 机器的 CPU。
	"/gpu:0": 机器的第一个 GPU, 如果有的话。
	"/gpu:1": 机器的第二个 GPU, 以此类推。

交互式使用
	为了便于使用诸如 IPython 之类的 Python 交互环境.
	可以使用 InteractiveSession 代替 Session 类， 使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run()。
	这样可以避免使用一个变量来持有会话。
'''
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
print sub.eval()

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session():
  result = sess.run([mul, intermed])
  print result

"""
Feed
	TensorFlow 提供了 feed 机制，该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁，直接插入一个 tensor。
	feed 只在调用它的方法内有效，方法结束，feed 就会消失。
	最常见的用例是将某些特殊的操作指定为 "feed" 操作，标记的方法是使用 tf.placeholder() 为这些操作创建占位符。

"""

