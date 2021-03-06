from ops import *
import timeit
from cifar10 import Cifar10


def convolute(x, filter_shape, input_size, output_size, cname):
	w = tf.Variable(tf.truncated_normal(shape=(filter_shape, filter_shape, input_size, output_size), mean = mu, stddev = sigma), name=cname+"_w") 
	b = tf.Variable(tf.zeros(output_size), name = cname+"_b")
	conv = tf.nn.conv2d(x, w, strides = [1,1,1,1], padding="VALID") + b
	conv = tf.nn.relu(conv)
	return conv

def fullycon(x,input_size, output_size ,cname):
  	w = tf.Variable(tf.truncated_normal(shape = (input_size, output_size), mean = mu, stddev = sigma), name =cname + '_w')
	b = tf.Variable(tf.zeros(output_size), name = cname + '_b')
	fl = tf.matmul(x, w) + b
	#activation function relu
	fl = tf.nn.relu(fl) #(1200)
    return fl


def net(input, is_training, dropout_kept_prob):
	
	#our network
	layers1 = 6
	for i in range(layers1):
		if i == 0:
			x = convolute(x,1,1,64, str(i+1))
			skip = x
		else:
			if (i % 2) == 0: # if an even convolution
				x = x + skip
				skip = x
			x = convolute(x,1,64,64, str(i+1))

	layers2 = 8
	for i in range(layers2):
		if i == 0:
			x = convolute(x, 1, 64, 128, str(i + 1 + layers1))
		else:
			if (i % 2) == 0: # if an even convolution
				x = x + skip
				skip = x
			x = convolute(x, 1, 128, 128, str(i + 1 + layers1))
	
	layers3 = 12
	for i in range(layers3):
		if i == 0:
			x = convolute(x, 1, 128, 256, str(i + 1 + layers1 + layers2))
		else:
			if (i % 2) == 0: # if an even convolution
				x = x + skip
				skip = x
			x = convolute(x, 1, 256, 256, str(i + 1 + layers1 + layers2))
	
	layers4 = 6
	for i in range(layers4):
		if i == 0:
			x = convolute(x, 1, 256, 512, str(i + 1 + layers1 + layers2 + layers3))
		else:
			if (i % 2) == 0: # if an even convolution
				x = x + skip
				skip = x
			x = convolute(x, 1, 512, 512, str(i + 1 + layers1 + layers2 + layers3))

	#FULLYCON
	x = flatten(x)
	x = fullycon(x, 4608, 1024,"fcon",True)
	x = fullycon(x, 1024, 256,"fcon",True)
	x = fullycon(x, 256, 64 ,"fcon",True)
	x = fullycon(x, 64 ,10,"fcon",False)


		
	return flf
		
def train():
	# Always use tf.reset_default_graph() to avoid error
	tf.reset_default_graph()
	# TODO: Write your training code here
	# - Create placeholder for inputs, training boolean, dropout keep probablity
	# - Construct your model
	# - Create loss and training op
	# - Run training
	# AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
	# YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
		# AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET
		raise NotImplementedError

def test(cifar10_test_images):
		# Always use tf.reset_default_graph() to avoid error
		tf.reset_default_graph()
		# TODO: Write your testing code here
		# - Create placeholder for inputs, training boolean, dropout keep probablity
		# - Construct your model
		# (Above 2 steps should be the same as in train function)
		# - Create label prediction tensor
		# - Run testing
		# DO NOT RUN TRAINING HERE!
		# LOAD THE MODEL AND RUN TEST ON THE TEST SET
		raise NotImplementedError