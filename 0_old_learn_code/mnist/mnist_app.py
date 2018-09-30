#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_SAVE_PATH = "./model/"
def restore_model(testPicArr):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		# 实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模块的更新速度，训练过程中对每一个变量维护一个影子变量，这个影子变量的初值
		# 就是相应变量的初始值，每次变量更新时，影子变量就会随之更新
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)


		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				preValue = sess.run(preValue, feed_dict = {x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	reIm.save("test.png")
	im_arr = np.array(reIm.convert('L'))
	threshold = 100
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			# print(im_arr[i][j])
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else: im_arr[i][j] = 255
	# print(im_arr)
	nm_arr = im_arr.reshape([1,784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)
	# print(img_ready)
	return img_ready

def application():
	testNum = input("请输入要识别的数字图片个数:")
	testNum = int(testNum)
	for i in range(testNum):
		testPic = input("请输入要识别的数字图片:")
		testPicArr = pre_pic(testPic)
		# print(testPicArr)
		preValue = restore_model(testPicArr)
		print("信不信由你，反正这个数字是:", preValue)
def main():
	application()

if __name__ == '__main__':
	main()