#coding:utf-8
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt #pyhton中输出结果可视化模块

#定义添加神经层的函数def add_layer(),它有四个参数：
#输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None
def add_layer(inputs,input_size,output_seiz,n_layer,activation_function = None):	
	layer_name = "layer%s" %n_layer
	with tf.name_scope("layer_name"):
		#开始定义weights和biases
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([input_size,output_seiz]),name = "W")#in_size行, out_size列的随机变量矩阵
			tf.summary.histogram(layer_name+"/Weights",Weights) #添加后可以观看变化的变量
		#机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1
		with tf.name_scope('Biases'):
			biases = tf.Variable(tf.zeros([1,output_seiz])+0.1,name = "b")
			tf.summary.histogram(layer_name+"/Biases",biases)
		#定义神经网络未激活的值。tf.matmul()是矩阵的乘法。
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs,Weights)+biases
			tf.summary.histogram(layer_name+"/Wx_plus_b",Wx_plus_b)
		#当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，
		#不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
		if activation_function is None:
			#　None 表示线性关系　无需激励函数
			output_seiz = Wx_plus_b
		else:
			output_seiz = activation_function(Wx_plus_b)

		# 返回输出 添加一个神经层的函数——def add_layer()就定义好了。
		tf.summary.histogram(layer_name+"/output_seiz",output_seiz) #添加后可以观看变化的变量
		return output_seiz


x_data = np.linspace(-1,1,300)[:,np.newaxis] #-1到１这个区间里有300个单位，维度 二维
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32) #噪点
y_data = np.square(x_data)-0.5 #x_data的二次放　－　０．５

#输入
with tf.name_scope("inputs_layer"):
	x_input = tf.placeholder(tf.float32,[None,1],name = "x_input")

	y_input = tf.placeholder(tf.float32,[None,1],name = "y_input")


#简单的三层神经网络：
#　输入层１个神经元　｜　隐藏层假设１０个神经元　｜　输出层１个神经元

#定义隐藏层
layer_hint = add_layer(x_input,1,10,n_layer = 1,activation_function = tf.nn.relu)

#定义输出层
layer_output = add_layer(layer_hint,10,1,n_layer =2,activation_function = None)
with tf.name_scope("loss"):
	loss =tf.reduce_mean(tf.reduce_sum(tf.square(y_input - layer_output),reduction_indices=[1]),name = "loss")#二者差的平方求和再取平均
	tf.summary.scalar("loss",loss)

#每一个练习的步骤　通过 优化器以0.1的学习效率对误差进行更正提升，下一次就会有更好的结果。
with tf.name_scope("train_step"):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始
init = tf.global_variables_initializer()

with tf.Session() as sess:
	#合并打包 summary 到文件中
	merged = tf.summary.merge_all()
	#把整个框架加载到文件中去，在通过浏览器观看这个文件 sess.graph 收集信息，并放在该文件中
	writer = tf.summary.FileWriter("logs/",sess.graph)
	sess.run(init)
	# #先生成图片框
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)#做连续性的画图
	# #plot上真实数据
	# ax.scatter(x_data,y_data)
	# plt.ion()
	# #打印输出
	# plt.show()
	for i in range(1000):
		sess.run(train_step, feed_dict={x_input:x_data,y_input:y_data})
		if i% 50:
			result = sess.run(merged,feed_dict={x_input:x_data,y_input:y_data}) #返回一个 summary 
			#将summary 放进 writer 里面
			writer.add_summary(result,i)
			# try:
			# 	#抹去上面一条线
			# 	ax.lines.remove(lines[0])
			# except Exception:
			# 	pass
			# #显示出预测数据
			# prediction_value = sess.run(layer_output,feed_dict={x_input:x_data})
			# #通过曲线形式plot上去 宽度为5的红色的线
			# lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
			# #暂停0.1秒
			# plt.pause(0.1)

			# print sess.run(loss,feed_dict={x_input:x_data,y_input:y_data})



