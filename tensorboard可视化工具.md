### 前言

&emsp;&emsp;tensorflow自带的可视化工具，tensorboard，只需要几个简单的操作就能够直观的展现tensorflow的数据流程。

### 样例

```python
import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
    # 该命令用来定义一个名称空间节点（op），其中的tf.Variable都要带上name属性
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/withgts',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='x_input')
    ys = tf.placeholder(tf.float32,[None, 1],name='y_input')

l1 = add_layer(xs,1,10,n_layer=1,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer=2,activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
    tf.summary.scalar('loss',loss)
    # tf.summary.scalar(tags, values, collections=None, name=None)用来显示标量信息，多用于显示损失函数loss和准确率accuary
    # 关于summary相关的拓展，见文末附录
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    # 将所有summary保存到磁盘中，方便tensorboard显示
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init_op)
    for x in range(10001):
        sess.run(train_step, feed_dict = {xs:x_data,ys:y_data})
        if x % 500 == 0:
            # print(sess.run(loss, feed_dict = {xs:x_data,ys:y_data}))
            result = sess.run(merged, feed_dict = {xs:x_data,ys:y_data})
            # 每500步，保存一次全部训练结果
            writer.add_summary(result,x)
            # FileWriter调用其add_summary()方法将训练过程数据保存在filewriter指定的文件中
```

### 附录

```python
# tf.summary.histogram(tags, values, collections=None, name=None) 用来显示直方图信息，多用于显示训练过程中的变量值分布
# tf.summary.distribution 分布图，一般用于显示weights分布
# tf.summary.text 可以将文本类型的数据转换为tensor写入summary中
# tf.summary.image(tag, tensor, max_images=3, collections=None, name=None) 输出带图像的probuf
# tf.summary.audio 展示训练过程中记录的音频
# tf.summary.merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了
# tf.summary.FileWritter(path,sess.graph) 指定一个文件用来保存图
# tf.summary.merge(inputs, collections=None, name=None) 一般选择要保存的信息还需要用到tf.get_collection()函数
```