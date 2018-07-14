import os
import tensorflow as tf
import numpy as np
from tensormodel import *
from Cifar_dataset import *

first_batch_size = 1000
num_batches = 5
batch_size = 1

model_path = os.path.dirname(__file__) + "\\model\\classifier.ckpt"
checkpoint_path = os.path.dirname(__file__) + "\\model\\checkpoint"
data_path = os.path.dirname(__file__) + '\\cifar-10-batches-py'

graph = tf.Graph()
with graph.as_default():
	data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='data')
	labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
	y_normalized = tf.placeholder(tf.float32, shape=[None, 10], name='y_normalized')
	Training_status = tf.placeholder(tf.bool)
	
	with tf.device("/device:GPU:0"):
		processed_images = pre_process_images(data, Training_status)
		
		W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,64],mean=0.0,stddev=0.163), name='W_conv1')
		b_conv1 = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv1')
		h_conv1 = tf.nn.relu(tf.nn.conv2d(processed_images, W_conv1, strides=[1,1,1,1], padding='SAME', name='conv1') + b_conv1)
		h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

		W_conv2 = tf.Variable(tf.truncated_normal([5,5,64,128],mean=0.0,stddev=0.035), name='W_conv2')
		b_conv2 = tf.Variable(tf.constant(0.005, shape=[128]), name='b_conv2')
		h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1], padding='SAME', name='conv2') + b_conv2)
		h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

		W_conv3A = tf.Variable(tf.truncated_normal([5,5,128,64],mean=0.0,stddev=0.025), name='W_conv3A')
		b_conv3A = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3A')
		h_conv3A = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3A, strides=[1,1,1,1], padding='SAME', name='conv3A') + b_conv3A)
		h_pool3A = tf.nn.max_pool(h_conv3A, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3A')

		W_conv3B = tf.Variable(tf.truncated_normal([4,4,128,64],mean=0.0,stddev=0.03125), name='W_conv3B')
		b_conv3B = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3B')
		h_conv3B = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3B, strides=[1,1,1,1], padding='SAME', name='conv3B') + b_conv3B)
		h_pool3B = tf.nn.max_pool(h_conv3B, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3B')

		W_conv3C = tf.Variable(tf.truncated_normal([3,3,128,64],mean=0.0,stddev=0.042), name='W_conv3B')
		b_conv3C = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3B')
		h_conv3C = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3C, strides=[1,1,1,1], padding='SAME', name='conv3B') + b_conv3C)
		h_norm3C = tf.layers.batch_normalization(h_conv3C, axis=1, training=Training_status)
		h_pool3C = tf.nn.max_pool(h_norm3C, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3C')

		W_conv3D = tf.Variable(tf.truncated_normal([2,2,128,64],mean=0.0,stddev=.0625), name='W_conv3D')
		b_conv3D = tf.Variable(tf.constant(0.005, shape=[64]), name='b_conv3D')
		h_conv3D = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3D, strides=[1,1,1,1], padding='SAME', name='conv3D') + b_conv3D)
		h_pool3D = tf.nn.max_pool(h_conv3D, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3D')

		h_AB_linked = tf.stack([h_pool3A, h_pool3B], axis=1)
		h_CD_linked = tf.stack([h_pool3C, h_pool3D], axis=1)
		h_linked_ = tf.stack([h_AB_linked, h_CD_linked], axis=1)
		h_linked = tf.reshape(h_linked_, [-1, 4*4*4*64])

		W_fc = tf.Variable(tf.truncated_normal([4*4*4*64, 1024],mean=0.0,stddev=0.022))
		b_fc = tf.Variable(tf.constant(0.005, shape=[1024]))
		h_fc = tf.nn.relu(tf.matmul(h_linked, W_fc) + b_fc)
		h_fc_norm = tf.layers.batch_normalization(h_fc, axis=1, training=Training_status)
		
		h_fc_drop = tf.layers.dropout(h_fc_norm, rate=0.5, training=Training_status)

		W_read = tf.Variable(tf.truncated_normal([1024,10],mean=0.0,stddev=0.0442))
		b_read = tf.Variable(tf.constant(0.005, shape=[10]))
		y_conv = tf.matmul(h_fc_drop, W_read) + b_read
		
	#Optimize, calculate accuracy and get the class prediction percentages as a set of normalized vectors
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y_conv))
	Optimize = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	y_normalized = tf.nn.softmax(y_conv)
	
	saver = tf.train.Saver()
	init = tf.global_variables_initializer()
	
	
with tf.Session(graph=graph) as sess:
	sess.run(init)
	if os.path.isfile(checkpoint_path):
		saver.restore(sess, model_path)
		print("Model restored.")
	else:
		print('No model found!')
	
	#Load cifar10 train dataset
	cifar = Cifar_dataset(data_path, training=False)
	
	#Long batch to see accuracy
	image_batch, hot_label_batch = cifar.get_random_batch(batch_size, one_hot=True)
	acc, y_norm = sess.run([accuracy, y_normalized], feed_dict={data: image_batch, labels: hot_label_batch, Training_status: False})
	print("Avg. Accuracy: "+str(acc))
	
	
	#Display y_norm layer for debugging purposes
	print(cifar.classes)
	np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
	for i in range(num_batches):
		image_batch, hot_label_batch = cifar.get_random_batch(batch_size, one_hot=True)
		acc, y_norm = sess.run([accuracy, y_normalized], feed_dict={data: image_batch, labels: hot_label_batch, Training_status: False})
		
		#Display confidences for each class and display image w/ actual class
		print("Percents: "+str(y_norm[0]))
		cifar.display_batch_first()
		