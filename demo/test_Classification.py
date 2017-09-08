# coding:utf-8
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs,input_size,output_seiz,n_layer,activation_function = None):	
	layer_name = "layer%s" %n_layer
	with tf.name_scope("layer_name"):
		#开始定义weights和biases
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([input_size,output_seiz]),name = "W")
		with tf.name_scope('Biases'):
			biases = tf.Variable(tf.zeros([1,output_seiz])+0.1,name = "b")
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs,Weights)+biases
		if activation_function is None:
			output_seiz = Wx_plus_b
		else:
			output_seiz = activation_function(Wx_plus_b)
		return output_seiz

def compute_accuracy(v_x,v_y):
	global layer_hint
	y_pre = sess.run(layer_hint,feed_dict={x_input:v_x})#概率
	cross_prediction =tf.equal(tf.argmax(y_pre,1),tf.argmax(v_y,1))#预测概率最大值和真实数据的差别
	accuracy = tf.reduce_mean(tf.cast(cross_prediction,tf.float32))#计算这组数据有多少对的多少错的
	reslut = sess.run(accuracy,feed_dict = {x_input:v_x,y_input:v_y})#输出 百分比
	return reslut

#输入
with tf.name_scope("inputs_layer"):
	x_input = tf.placeholder(tf.float32,[None,784])#28x28

	y_input = tf.placeholder(tf.float32,[None,10])


#简单的三层神经网络：
#　输入层１个神经元　｜　隐藏层假设１０个神经元　｜　输出层１个神经元

#定义隐藏层
layer_hint = add_layer(x_input,784,10,n_layer = 1,activation_function = tf.nn.softmax)

#定义输出层
layer_output = add_layer(layer_hint,10,1,n_layer =2,activation_function = None)
#及算损失
with tf.name_scope("loss"):
	cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_input *tf.log(layer_hint),reduction_indices=[1]))#二者差的平方求和再取平均


#每一个练习的步骤　通过 优化器以0.1的学习效率对误差进行更正提升，下一次就会有更好的结果。
with tf.name_scope("train_step"):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#初始
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for i in range(1000):
		#提取出一部分的数据
		batch_x_input ,batch_y_input= mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x_input:batch_x_input,y_input:batch_y_input})
		if i% 400:
			print compute_accuracy(mnist.test.images, mnist.test.labels)
			



