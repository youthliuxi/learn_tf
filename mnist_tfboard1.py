# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
max_step=1000
learning_rate=0.001
dropout=0.9
data_dir='MNIST_data'
log_dir='tfboard3'
mnist=input_data.read_data_sets(data_dir,one_hot=True)
sess=tf.InteractiveSession()

with tf.name_scope('input'):
	x=tf.placeholder(tf.float32,[None,784],name='X_input')
	y=tf.placeholder(tf.float32,[None,10],name='y_input')
with tf.name_scope('input_reshape'):
	image_shaped_input=tf.reshape(x,[-1,28,28,1])
	tf.summary.image('input',image_shaped_input)

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def variable_summaries(var):
	with tf.name_scope('summeries'):
		mean=tf.reduce_mean(var)
		tf.summary.scalar('mean',mean)
		with tf.name_scope('stddev'):
			stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev',stddev)
		tf.summary.scalar('max',tf.reduce_max(var))
		tf.summary.scalar('min',tf.reduce_min(var))
		tf.summary.histogram('histogram',var)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weight'):
			weights=weight_variable([input_dim,output_dim])
			variable_summaries(weights)
		with tf.name_scope('biases'):#same layer ,all iamges share one biase
			biases=bias_variable([output_dim])
			variable_summaries(biases)
		with tf.name_scope('wx_plus_b'):
			preactivate=tf.matmul(input_tensor,weights)+biases
			tf.summary.histogram('preactivate',preactivate)
		activations=act(preactivate,name='activate')
		return activations #not really calcute
hidden1=nn_layer(x,784,500,'layer1')

with tf.name_scope('dropout'):
	keep_prob=tf.placeholder(tf.float32)
	tf.summary.scalar('keep_prob',keep_prob)
	droped=tf.nn.dropout(hidden1,keep_prob)
	#calcute relu,print(name,value) to see
y1=nn_layer(droped,500,10,'layer2',act=tf.identity)

with tf.name_scope('cross_entropy'):
	diff=tf.nn.softmax_cross_entropy_with_logits(logits=y1,labels=y)
	with tf.name_scope('total'):
		cross_entropy=tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy_prediction'):
	correct_prediction=tf.equal(tf.argmax(y1,1),tf.argmax(y,1))
	with tf.name_scope('accuracy'):
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()		#learning this write way
train_writer=tf.summary.FileWriter(log_dir+'/trian',sess.graph)
test_writer=tf.summary.FileWriter(log_dir+'/test')#save events,but not tfboard
tf.global_variables_initializer().run()

def feed_dict(train):
	if train:
		xs,ys=mnist.train.next_batch(100)
		k=dropout
	else:
		xs,ys=mnist.test.images,mnist.test.labels
		k=1.0#key=placeholder_name ,learn 3 elements
	return {x:xs,y:ys,keep_prob:k}

saver=tf.train.Saver()
for i in range(max_step):
	if i%10==0:
		summary,acc=sess.run([merged,accuracy],feed_dict=feed_dict(False))
		test_writer.add_summary(summary,i)#learn %s can value int,float
		print('accuracy at step %s:%s' %(i,acc))
	else:
		if i%100==99:#???
			run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata=tf.RunMetadata()#???
			summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True),options=run_options,run_metadata=run_metadata)
			train_writer.add_run_metadata(run_metadata,'step%03d' %i)#??
			train_writer.add_summary(summary,i)
			saver.save(sess,log_dir+'/model.ckpt',global_step=i)
			print('adding run metadata for %s' %i)
		else:
			summary,_=sess.run([merged,train_step],feed_dict=feed_dict(True))
			train_writer.add_summary(summary,i)
train_writer.close()
test_writer.close()