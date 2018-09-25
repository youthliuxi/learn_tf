## tensorflow代码初识
&emsp;&emsp;tensor张量，flow流动，其思想就是构建一张记录张量流动方向的网，然后，激活该网络，同时给定输入值，从而不断矫正网络中需要修正的参数W、b，也就是所谓的训练。
&emsp;&emsp;部分源码解读（以注释的形式）
#### 简单的线性函数训练（python1.py）
```python
import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x_data = np.random.rand(100).astype(np.float32)
# 使用numpy模块生成一个格式为32浮点型的100以内的随机数
# tf.random_uniform((4, 4), minval=-0.5,maxval=0.5,dtype=tf.float32)
# 更完整的随机数的例子
y_data = x_data*0.5 + 1
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# 将Weight定义为tensorflow的变量，同时赋予初值（一个一维随机数，范围是-1到1）
# 与python原生不同的是，tensorflow只有定义为变量才会被认定为变量，只要程序中出现变量，就一定要初始化。
biases = tf.Variable(tf.zeros([1]))

y = Weight*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
# tf.square计算平方
# loss = tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 计算平均值（输入张量，在哪个唯独求均值，keep_dims=False，name=None）
# reduction_indices不填时，对张量所有值求平均，0时对每一列求平均值得到一个行向量，1时对每一行求平均，得到一个列向量。

optimizer = tf.train.GradientDescentOptimizer(0.5)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate, use_locking=False,name=’GradientDescent’)
# 实现梯度下降算法的优化器，通常只需要输入学习速率即可，其他参数为默认选项
# 还有更多优化器等待学习
train = optimizer.minimize(loss)
# optimizer变量保存上述梯度下降优化器，loss变量保存均方误差，此句本意：使用梯度下降优化器使均方误差达到最小值
# 这是构建了一个训练点，用来训练optimizer优化器
# init = tf.initialize_all_variables() 老版本的代码，现已不再使用
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # 激活session
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
# 使用session，初始化所有参数
for step in range(201):
    sess.run(train)
 if step %20 == 0:
     print(step,sess.run(Weight),sess.run(biases))
sess.close()
```
#### 变量与常量的样例（python3.py）
```python
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
state = tf.Variable(0, name = 'counter')
# 定义变量state，赋值为0，并定义了一个名字叫：counter
print(state.name)
one = tf.constant(1)
# 定义一个常量1
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# 将new_value的值重新赋值给state
init = tf.global_variables_initializer()
# 初始化所有变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```
#### 输入样例（python4.py）
```python
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# input1 = tf.placeholder(tf.float32, [2,2])
# 上述是规定了输入结构为2行2列的张量
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
# 定义输入和输出
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:[7.0], input2:[2.0]})
    print(result)
```
#### 二次函数训练样例（python5.py）
```python
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
# # [2]
# # [3]
# # [4]
# # [5]
# # [6]]
```