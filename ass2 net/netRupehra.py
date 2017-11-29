from ops import *
from cifar10 import Cifar10

cifar10_test = Cifar10(test=True, shuffle=False, one_hot=False)
cifar10_test_images, cifar10_test_labels = cifar10_test._images, cifar10_test._labels

def net(input, is_training, dropout_kept_prob):

	xavier = tf.contrib.layers.xavier_initializer()

	#TODO: Convolution 1
	conv1_W = tf.get_variable("conv1_W", shape=[5,5,3,16],initializer=xavier)
	conv1_b = tf.Variable(tf.zeros(16))
	convolve_layer1= tf.nn.conv2d(input, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

	batch_norm_layer = get_batch_norm(convolve_layer1,is_training,name="batch_norm1")
	relu_layer = tf.nn.relu(batch_norm_layer)
	max_pool_layer1 = tf.nn.max_pool(relu_layer, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding="SAME")

	#TODO: Convolution 2
	conv2_W = tf.get_variable("conv2_W", shape=[5, 5, 16, 32], initializer=xavier)
	conv2_b = tf.Variable(tf.zeros(32))
	convolve_layer2 = tf.nn.conv2d(max_pool_layer1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

	batch_norm_layer2 = get_batch_norm(convolve_layer2,is_training,name="batch_norm2")
	relu_layer2 = tf.nn.relu(batch_norm_layer2)
	max_pool_layer2 = tf.nn.max_pool(relu_layer2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

	#TODO: Convolution 3
	conv3_W = tf.get_variable("conv3_W", shape=[5, 5, 32, 64], initializer=xavier)
	conv3_b = tf.Variable(tf.zeros(64))
	convolve_layer3 = tf.nn.conv2d(max_pool_layer2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

	batch_norm_layer3 = get_batch_norm(convolve_layer3,is_training,name="batch_norm3")
	relu_layer3 = tf.nn.relu(batch_norm_layer3)
	max_pool_layer3 = tf.nn.max_pool(relu_layer3,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")

	conv4_W = tf.get_variable("conv4_W", shape=[1,1,64,64], initializer=xavier)
	conv4_b = tf.Variable(tf.zeros(64))
	convolve_layer4 = tf.nn.conv2d(max_pool_layer3,conv4_W, strides=[1,1,1,1], padding='SAME') + conv4_b

	batch_norm_layer4 = get_batch_norm(convolve_layer4, is_training, name="batch_norm4")
	relu_layer4 = tf.nn.relu(batch_norm_layer4)

	conv5_W = tf.get_variable("conv5_W", shape=[1,1,64,64], initializer=xavier)
	conv5_b = tf.Variable(tf.zeros(64))
	convolve_layer5 = tf.nn.conv2d(relu_layer4,conv4_W, strides=[1,1,1,1], padding='SAME') + conv5_b

	batch_norm_layer5 = get_batch_norm(convolve_layer5, is_training, name="batch_norm5")
	relu_layer5 = tf.nn.relu(batch_norm_layer5)

	skip_layer = max_pool_layer3 + relu_layer5

	flatten_max_pool_layer = tf.contrib.layers.flatten(skip_layer)

	# #TODO: Fully connected layer 1
	fc1_W = tf.get_variable("fc1_W", shape=[1024,512], initializer=xavier)
	fc1_b = tf.Variable(tf.zeros(512))
	fc1= tf.matmul(flatten_max_pool_layer,fc1_W) + fc1_b
	fc1 = tf.nn.relu(fc1)

	# TODO: Fully connected layer 2
	fc2_W = tf.get_variable("fc2_W", shape=[512,256], initializer=xavier)
	fc2_b = tf.Variable(tf.zeros(256))
	fc2 = tf.matmul(fc1, fc2_W) + fc2_b
	fc2 = tf.nn.relu(fc2)

	#TODO: Dropout
	fc2 = tf.contrib.layers.dropout(fc2,keep_prob=1-dropout_kept_prob,is_training=is_training)

	#TODO: Fully connected layer 3
	fc3_W = tf.get_variable("fc3_W", shape=[256,128], initializer=xavier)
	fc3_b = tf.Variable(tf.zeros(128))
	fc3 = tf.matmul(fc2, fc3_W) + fc3_b
	fc3 = tf.nn.relu(fc3)

	# TODO: Fully connected layer 4
	fc4_W = tf.get_variable("fc4_W",shape=[128,10], initializer=xavier)
	fc4_b = tf.Variable(tf.zeros(10))
	logits = tf.matmul(fc3, fc4_W) + fc4_b

	return logits

def train():
	# Always use tf.reset_default_graph() to avoid error
	tf.reset_default_graph()

	#Gettting the training set
	cifar10_train = Cifar10(test=False, shuffle=False, one_hot=True)
	x_train, y_train = cifar10_train._images, cifar10_train._labels
	cifar10_train_images = tf.cast(x_train, tf.float32)
	number_of_images = cifar10_train_images.shape[0]

	# - Create placeholder for inputs, training boolean, dropout keep probablity
	x = tf.placeholder(tf.float32, (None, 32, 32, 3))
	y = tf.placeholder(tf.int32, (None,10))
	# one_hot_y = tf.one_hot(y, 10)
	is_training = True
	dropout_keep_probability = 0.2

	rate = tf.placeholder(tf.float32)
	# - Construct your model
	logits = net(x, is_training, dropout_keep_probability)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

	# - Create loss and training op
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = rate)
	grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	training_operation = optimizer.apply_gradients(grads_and_vars)
	lr = 0.001

	n_epochs = 15 # if n_epochs = 10, takes 10 minutes to train
	batch_size = 250
	saver = tf.train.Saver()

	# - Run training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			# print("epoch: ", epoch)
			n_batches = number_of_images // batch_size
			start = 0
			end = batch_size
			if epoch % 2 == 0 and epoch != 0:
				lr = lr/2

			for iteration in range(n_batches):
				if iteration % 10 == 0:
					print("iteration: ", iteration, " start: ", start, " end: ", end)
				batch_x, batch_y = x_train[start:end], y_train[start:end]
				sess.run([training_operation],feed_dict={x: batch_x, y: batch_y,rate:lr })
				start = end
				end += batch_size
			saver.save(sess,'ckpt/netModel')
				# raise NotImplementedError

def test(cifar10_test_images):
	tf.reset_default_graph()
	# Gettting the testing set
	cifar10_train = Cifar10(test=True, shuffle=False, one_hot=True)
	x_test, y_test = cifar10_train._images, cifar10_train._labels
	cifar10_train_images = tf.cast(x_test, tf.float32)
	number_of_images = cifar10_train_images.shape[0]

	# - Create placeholder for inputs, training boolean, dropout keep probablity
	x = tf.placeholder(tf.float32, (None, 32, 32, 3))
	y = tf.placeholder(tf.int32, (None,10))
	is_training = False
	dropout_keep_probability = 1
	n_epochs = 1  # if n_epochs = 10, takes 10 minutes to train
	batch_size = 250

	# - Construct your model
	logits = net(x, is_training, dropout_keep_probability)
	training_yp = tf.argmax(logits,1)

	# - Create label prediction tensor
	Y = []
	saver = tf.train.Saver()

	# - Run testing
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('ckpt'))

		for epoch in range(n_epochs):
			n_batches = number_of_images // batch_size
			start = 0
			end = batch_size
			for iteration in range(n_batches):
				# print("iteration: ", iteration, " start: ", start, " end: ", end)
				batch_x, batch_y = x_test[start:end], y_test[start:end]
				predicted_y = sess.run([training_yp], feed_dict={x: batch_x, y: batch_y})
				start = end
				end += batch_size

				if len(Y):
					Y = np.hstack((Y,predicted_y))
				else:
					Y = predicted_y
	return Y
