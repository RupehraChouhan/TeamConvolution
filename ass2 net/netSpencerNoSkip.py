from ops import *
import timeit
from cifar10 import Cifar10
import tensorflow as tf
import numpy as np

#batch 125, lr = 0.003, epochs 15 = 68% with skip
#batch 125, lr = 0.004, epochs 15 = 67% with skip

BATCH_SIZE = 100

def net(input, is_training, dropout_kept_prob):
	# TODO: Write your network architecture here
	# Or you can write it inside train() function
	# Requirements:
	# - At least 5 layers in total
	# - At least 1 fully connected and 1 convolutional layer
	# - At least one maxpool layers
	# - At least one batch norm
	# - At least one skip connection
	# - Use dropout

	# Hyperparameters
	mu = 0
	sigma = 0.1
	

	xavier = tf.contrib.layers.xavier_initializer()


	# Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
	#conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma), name='conv1_W')
	conv1_W = tf.get_variable("conv1_W", shape=[5, 5, 3, 16], initializer=xavier)
	conv1_b = tf.Variable(tf.zeros(16), name='conv1_b')
	conv1   = tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
	drop = tf.layers.dropout(conv1,rate=1-dropout_kept_prob, noise_shape=None, seed=None, training=is_training, name='dropout1')


	# Layer 1: Batch Normalization
	batch_norm1 = batch_norm(drop, is_training, name="batch_norm1")
	# Layer 1: Activation
	activation1 = tf.nn.relu(batch_norm1)

	# Layer 1: Pooling. Input = 28x28x6. Output = 14x14x6.
	# print(activation1.shape) # (?, 32, 32, 32)
	pool1 = tf.nn.max_pool(activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# print(pool1.shape) # (?, 16, 16, 32)




	# Layer 2: Convolutional. Input = 28x28x6. Output = 24x24x10.
	#conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name='conv2_W')
	conv2_W = tf.get_variable("conv2_W", shape=[5, 5, 16, 32], initializer=xavier)
	conv2_b = tf.Variable(tf.zeros(32), name='conv2_b')
	conv2   = tf.nn.conv2d(pool1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
	drop = tf.layers.dropout(conv2,rate=1-dropout_kept_prob, noise_shape=None, seed=None, training=is_training, name='dropout2')


	# Layer 2: Batch Normalization
	batch_norm2 = batch_norm(drop, is_training, name="batch_norm2")
	# Layer 2: Activation.
	activation2 = tf.nn.relu(batch_norm2)
	pool2 = tf.nn.avg_pool(activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




	# Layer 3: Convolutional. Input = 24x24x16. Output = 20x20x24.
	#conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 6), mean = mu, stddev = sigma), name='conv3_W')
	conv3_W = tf.get_variable("conv3_W", shape=[5, 5, 32, 64], initializer=xavier)
	conv3_b = tf.Variable(tf.zeros(64), name='conv3_b')
	conv3   = tf.nn.conv2d(pool2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
	drop = tf.layers.dropout(conv3,rate=1-dropout_kept_prob, noise_shape=None, seed=None, training=is_training, name='dropout3')


	# Layer 3: Batch Normalization
	batch_norm3 = batch_norm(conv3, is_training, name="batch_norm3")

	activation3 = tf.nn.relu(batch_norm3)

	pool3 = tf.nn.avg_pool(activation3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	print(pool3.shape)



	# Skip
	flat = tf.contrib.layers.flatten(pool1)
	fc_skipW = tf.get_variable("fc_skip", shape=[4096, 1024], initializer=xavier)
	fc_skipb = tf.Variable(tf.zeros(1024), name='fc_skipb')
	flat = tf.matmul(flat, fc_skipW) + fc_skipb
	unflat = tf.reshape(flat, [BATCH_SIZE, 4, 4, 64])
	skip = unflat + pool3

	# Layer 3: Activation.
	#activation3 = tf.nn.relu(skip)
	# Layer 3: Pooling. Input = 10x10x16. Output = 5x5x16.
	#pool3 = tf.nn.max_pool(activation3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	# Flatten. Input = 14x14x6. Output = 400.
	fc0   = tf.contrib.layers.flatten(pool3)

	
	# Layer 3: Fully Connected. Input = 400. Output = 120.
	#fc1_W = tf.Variable(tf.truncated_normal(shape=(1536, 800), mean = mu, stddev = sigma), name='fc1_W')
	fc1_W = tf.get_variable("fc1_W", shape=[1024, 512], initializer=xavier)
	fc1_b = tf.Variable(tf.zeros(512), name='fc1_b')
	fc1   = tf.matmul(fc0, fc1_W) + fc1_b
	# Activation.
	fc1    = tf.nn.relu(fc1)

	#dropout1 = tf.layers.dropout(fc1,rate=1-dropout_kept_prob, noise_shape=None, seed=None, training=False, name='dropout1')

	# Layer 4: Fully Connected. Input = 120. Output = 84.
	#fc2_W  = tf.Variable(tf.truncated_normal(shape=(800, 300), mean = mu, stddev = sigma), name='fc2_W')
	fc2_W = tf.get_variable("fc2_W", shape=[512, 256], initializer=xavier)
	fc2_b  = tf.Variable(tf.zeros(256), name='fc2_b')
	fc2    = tf.matmul(fc1, fc2_W) + fc2_b
	# Activation.
	fc2    = tf.nn.relu(fc2)

	dropout2 = tf.layers.dropout(fc2,rate=1-dropout_kept_prob, noise_shape=None, seed=None, training=False, name='dropout4')

	# Layer 5: Fully Connected. Input = 84. Output = 10.
	#fc3_W  = tf.Variable(tf.truncated_normal(shape=(300, 100), mean = mu, stddev = sigma), name='fc3_W')
	fc3_W = tf.get_variable("fc3_W", shape=[256, 10], initializer=xavier)
	fc3_b  = tf.Variable(tf.zeros(10), name='fc3_b')
	logits = tf.matmul(dropout2, fc3_W) + fc3_b


	# fc4_W  = tf.Variable(tf.truncated_normal(shape=(100, 10), mean = mu, stddev = sigma), name='fc4_W')
	# fc4_b  = tf.Variable(tf.zeros(10), name='fc4_b')
	# logits = tf.matmul(fc3, fc4_W) + fc4_b
	
	return logits

def add_gradient_summaries(grads_and_vars):
	for grad, var in grads_and_vars:
		if grad is not None:
			tf.summary.histogram(var.op.name + "/gradient", grad)

def train():
	# Always use tf.reset_default_graph() to avoid error
	# TODO: Write your training code here
	# - Create placeholder for inputs, training boolean, dropout keep probablity
	# - Construct your model
	# - Create loss and training op
	# - Run training
	# AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
	# YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
	# AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET


	tf.reset_default_graph()

	#tf.reset_default_graph()
	x = tf.placeholder(tf.float32, (None, 32, 32, 3))
	y = tf.placeholder(tf.int32, (None, 10))
	is_training = True;
	dropout_keep_probablity = 0.95

	rate = 0.001
	EPOCHS = 15


	cifar10_train = Cifar10(batch_size=BATCH_SIZE, one_hot=True, test=False, shuffle=False)
	cifar10_train_images, cifar10_train_labels = cifar10_train._images, cifar10_train._labels  # Get all images and labels of the test set.
	cifar10_train_images = tf.cast(cifar10_train_images, tf.float32)
	#print(batch_y.shape);
	logits = net(x, is_training, dropout_keep_probablity);

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	training_operation = optimizer.apply_gradients(grads_and_vars)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name + "/histogram", var)

	add_gradient_summaries(grads_and_vars)
	tf.summary.scalar('loss_operation', loss_operation)
	merged_summary_op= tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('logs/')


	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#sess.run(conv1_W.initializer)
		num_examples = cifar10_train_images.get_shape().as_list()[0]
		global_total = EPOCHS*(num_examples/BATCH_SIZE)
		
		#training_operation, merged_summary_op = train_lenet()
		print("Training...")
		global_step = 0
		for i in range(EPOCHS):
			#cifar10_train_images, cifar10_train_labels = shuffle(cifar10_train_images, cifar10_train_labels)
			for offset in range(0, num_examples, BATCH_SIZE):
				print(str(global_step/global_total*100) + "%" + " done training")
				#print(offset)
				batch_x, batch_y = cifar10_train.get_next_batch()
				_, summaries = sess.run([training_operation, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
				if global_step % 100 == 1:
					summary_writer.add_summary(summaries, global_step=global_step)
				global_step += 1
				
			saver.save(sess, 'ckpt/spencenet', global_step=i)
			rate /= 2 # anneal learning rate
			print("Learning rate: " + str(rate))
			
		print("Model saved")

	sess.close()

	#raise NotImplementedError

def test(cifar10_test_images):
	# Always use tf.reset_default_graph() to avoid error
	# TODO: Write your testing code here
	# - Create placeholder for inputs, training boolean, dropout keep probablity
	# - Construct your model
	# (Above 2 steps should be the same as in train function)
	# - Create label prediction tensor
	# - Run testing
	# DO NOT RUN TRAINING HERE!
	# LOAD THE MODEL AND RUN TEST ON THE TEST SET
	tf.reset_default_graph()

	#tf.reset_default_graph()
	x = tf.placeholder(tf.float32, (None, 32, 32, 3))
	y = tf.placeholder(tf.int32, (None, 10))
	is_training = False;
	dropout_keep_probablity = 1

	EPOCHS = 1

	cifar10_test = Cifar10(batch_size=BATCH_SIZE, one_hot=True, test=True, shuffle=False)
	cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels  # Get all images and labels of the test set.
	cifar10_test_images = tf.cast(cifar10_test_images, tf.float32)
	#print(cifar10_train_labels.shape);
	#print(batch_y.shape);
	logits = net(x, is_training, dropout_keep_probablity);
	training_op = tf.argmax(logits, 1)


	predicted_y_test = []


	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
		num_examples = cifar10_test_images.shape[0]
		#sess.run(tf.global_variables_initializer())
		print("Testing...")
		global_step = 0
		for i in range(EPOCHS):
			#X_train, y_train = shuffle(X_train, y_train)
			for offset in range(0, num_examples, BATCH_SIZE):
				batch_x, batch_y = cifar10_test.get_next_batch()
				yp = sess.run([training_op], feed_dict={x: batch_x, y: batch_y})
				global_step += 1
				
				if (len(predicted_y_test) == 0):
					predicted_y_test = yp
				else:
					predicted_y_test = np.hstack((predicted_y_test, yp))



	sess.close()
	return predicted_y_test


if __name__ == '__main__':
	train();
