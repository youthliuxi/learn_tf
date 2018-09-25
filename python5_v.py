import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_layer(inputs,in_size,out_size,activation_function = None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs
# 自定义构造神经网络层的函数，无论是输入层、隐藏层还是输出层都可以使用该函数进行定义
# 每一层都是最简单的wx+b的形式
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
# 构造y值的噪声，随机波动
y_data = np.square(x_data) - 0.5 + noise
# 构造y值，并加入随机波动

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 输入预留

l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 使用优化器优化loss值
init_op = tf.global_variables_initializer()
# 初始化
with tf.Session() as sess:
	sess.run(init_op)
	# 输出真实数据 start
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data,y_data)
	plt.ion()
	plt.show()
	# plt.show(block=False)
	# 输出真实数据 end
	for x in range(10001):
		sess.run(train_step, feed_dict = {xs:x_data,ys:y_data})
		if x % 100 == 0:
			print(sess.run(loss, feed_dict = {xs:x_data,ys:y_data}))
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			# 可视化输出预测函数 start
			prediction_value = sess.run(prediction,feed_dict={xs:x_data})
			lines = ax.plot(x_data,prediction_value,'r-',lw=5)
			# 以红线描绘出训练后的结果
			plt.pause(0.1)
			# 输出间隔0.1秒
# 一、
# np.newaxis的功能是插入新的维度，举个例子
# a = np.array([1, 2, 3, 4, 5, 6])
# print(a)
# # [1 2 3 4 5 6]
# b=a[np.newaxis,:]
# print(b)
# # [[1 2 3 4 5 6]]
# c=a[:,np.newaxis]
# print(c)
# # [[1]
# #  [2]
# #  [3]
# #  [4]
# #  [5]
# #  [6]]