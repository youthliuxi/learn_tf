import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.5 + 1

Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weight*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
# # 初始化结构
sess = tf.Session()
sess.run(init) # 激活session
# with tf.Session() as sess:
# 	init_op = tf.global_variables_initializer()
# 	sess.run(init_op)
for step in range(201):
	sess.run(train)
	if step %20 == 0:
		print(step,sess.run(Weight),sess.run(biases))

sess.close()