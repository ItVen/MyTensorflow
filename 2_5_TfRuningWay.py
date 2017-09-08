#-*- coding:utf-8 -*-
'''
TensorFlow运作方式入门 
训练并评估一个用于识别手写数字的简易前馈神经网络
'''
from __future__ import division
import os.path
import time
import tensorflow as tf
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, '初始学习率。')
flags.DEFINE_integer('max_steps', 2000, '运行训练的步骤数。')
flags.DEFINE_integer('hidden1', 128, '隐藏层2的单位数。')
flags.DEFINE_integer('hidden2', 32, '隐藏层2的单位数。')
flags.DEFINE_integer('batch_size', 100, '批量大小。必须均匀地分成数据集大小。 ')
flags.DEFINE_string('train_dir', 'MNIST_data', '训练数据目录。')
flags.DEFINE_boolean('fake_data', False, '如果为真，则使用假数据进行单元测试。')



#--------- 输入与占位符-------------
def placeholder_inputs(batch_size):
	
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,mnist.IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	return images_placeholder, labels_placeholder
# ------向图表提供反馈--------
def fill_feed_dict(data_set, images_pl, labels_pl):
	'''
	以占位符为哈希键，创建一个Python字典对象，键值则是其代表的反馈Tensor。
	字典随后作为feed_dict参数，传入sess.run()函数中，为这一步的训练提供输入样例
	'''
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
	feed_dict = {
	  images_pl: images_feed,
	  labels_pl: labels_feed,
	}
	return feed_dict

# -------------------评估模型-------------
def do_eval(sess, eval_correct,  images_placeholder,  labels_placeholder,  data_set):
	true_count = 0  # 计算正确的预测数。 
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size
	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set,
		                           images_placeholder,
		                           labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	precision = true_count / num_examples
	print('  例子数: %d  正确数: %d  百分比: %0.04f' %
	    (num_examples, true_count, precision))

def run_training():
	data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
	# ----------图表 -----------------
	with tf.Graph().as_default():# with 这个命令表明所有已经构建的操作都要与默认的tf.Graph全局实例关联起来。
		images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
		# 建立一个从推理模型计算预测的图表。 
		logits = mnist.inference(images_placeholder, FLAGS.hidden1,FLAGS.hidden2)
	    #在图中添加损失计算的OPS。 
		loss = mnist.loss(logits, labels_placeholder)
	    # 在图中添加计算和应用渐变的OPS。
		train_op = mnist.training(loss, FLAGS.learning_rate)
	    # 在进入训练循环之前，我们应该先调用mnist.py文件中的evaluation函数，
		eval_correct = mnist.evaluation(logits, labels_placeholder)# 传入的logits和标签参数要与loss函数的一致。这样做事为了先构建Eval操作。

		# # evaluation函数会生成tf.nn.in_top_k 操作，如果在K个最有可能的预测中可以发现真的标签，
		# # 那么这个操作就会将模型输出标记为正确。在本文中，我们把K的值设置为1，
		# # 也就是只有在预测是真的标签时，才判定它是正确的。

		# eval_correct = tf.nn.in_top_k(logits, labels, 1)
		#  状态可视化 
		# 为了释放TensorBoard所使用的事件文件（events file），
		# 所有的即时数据（在这里只有一个）都要在图表构建阶段合并至一个操作（op）中。
		summary_op = tf.merge_all_summaries()
		# --------保存检查点（checkpoint）------------
		# 为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），我们实例化一个tf.train.Saver。
		saver = tf.train.Saver()	
		# 在图表上创建运行OPS的会话。 
		sess = tf.Session()
	    # 运行OP初始化变量。
		init = tf.initialize_all_variables()
		sess.run(init)
	    # 在创建好会话（session）之后，可以实例化一个tf.train.SummaryWriter，用于写入包含了图表本身和即时数据具体值的事件文件。
		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,graph_def=sess.graph_def)
		# 然后在一切都建立后，开始训练循环。
		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
			# 在代码中明确其需要获取的两个值：[train_op, loss]
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time
			if step % 100 == 0:
		        # 假设训练一切正常，没有出现NaN，训练循环会每隔100个训练步骤，就打印一行简单的状态文本，告知用户当前的训练状态。
				print('步骤 %d: 损失 = %.2f (%.3f 秒)' % (step, loss_value, duration))

		        # Update the events file.
		        # 最后，每次运行summary_op时，都会往事件文件中写入最新的即时数据，
				# 函数的输出会传入事件文件读写器（writer）的add_summary()函数。。

				summary_str = sess.run(summary_op, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)#summary_str是summary类型的，需要放入writer中，i步数（x轴） 

			# 每隔一千个训练步骤，我们的代码会尝试使用训练数据集与测试数据集，
			# 对模型进行评估。do_eval函数会被调用三次，分别使用训练数据集、验证数据集合测试数据集。
			if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				print "in"
				saver.save(sess, FLAGS.train_dir, global_step=step)
				print('训练数据的评价 :')
				do_eval(sess,
				    eval_correct,
				    images_placeholder,
				    labels_placeholder,
				    data_sets.train)
				# Evaluate against the validation set.
				print('验证数据评价:')
				do_eval(sess,
				    eval_correct,
				    images_placeholder,
				    labels_placeholder,
				    data_sets.validation)
				# Evaluate against the test set.
				print('测试数据评价:')
				do_eval(sess,
				    eval_correct,
				    images_placeholder,
				    labels_placeholder,
				    data_sets.test)



def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()

