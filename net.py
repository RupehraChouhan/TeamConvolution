from ops import *
import timeit
from cifar10 import Cifar10


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
  mu = 0
  sigma = 0.1
  
  # first convolution layer
  conv1_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 3, 80), mean = mu, stddev = sigma), name = 'conv1_w')
  conv1_w = tf.get_variable("conv1_w", shape = [5,5,3,80], initializer=tf.contrib.layers.xavier_initializer())
  conv1_b = tf.Variable(tf.zeros(80), name = 'conv1_b')
  conv1 = tf.nn.conv2d(input, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv1_b #(28,28,80)
  #activation function relu
  conv1 = tf.nn.relu(conv1)
  #norm1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'norm1')
  #max pooling
  conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')#(14,14,64)
  
  # second convolution layer
  convv_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 80, 160), mean = mu, stddev = sigma), name = 'convv_w')
  convv_b = tf.Variable(tf.zeros(160), name = 'convv_b')
  convv = tf.nn.conv2d(conv1, convv_w, strides = [1,1,1,1], padding = 'VALID') + convv_b #(10,10,160)  
  convv = batch_norm(convv, is_training, name = "bn_convv")
  convv = tf.nn.relu(convv) #(10,10,160)
  #print("shape of convv: ", convv)
  
  # second conv layer
  conv2_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 160, 160), mean = mu, stddev = sigma), name = 'conv2_w')
  conv2_b = tf.Variable(tf.zeros(160), name = 'conv2_b')
  conv2 = tf.nn.conv2d(convv, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b #(6,6,160)
  conv2 = batch_norm(conv2, is_training, name = "bn_conv2")
  #print("shape of conv2: ", conv2)
  #skip connection
      
    ##flatten convv to (?, 6, 6, 160)
  convv = tf.nn.max_pool(convv, ksize = [1, 5, 5, 1], strides = [1, 1, 1, 1], padding = 'VALID') #(6, 6, 160)
  conv2 = conv2 + convv
  #activation function relu
  conv2 = tf.nn.relu(conv2) #(10,10,160)
  conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID') #(5, 5, 160)
  #print("shape of conv2: ", conv2)
  
  # fully connected layer 1
  fl1 = flatten(conv2) #(1440)
  print(fl1)
  fl1_w = tf.Variable(tf.truncated_normal(shape = (1440, 400), mean = mu, stddev = sigma), name = 'fl1_w')
  fl1_b = tf.Variable(tf.zeros(400), name = 'fl1_b')
  fl2 = tf.matmul(fl1, fl1_w) + fl1_b
  #activation function relu
  fl2 = tf.nn.relu(fl2) #(1200)
  #print(fl2)
  
  '''fifth fully connected layer'''
  fl2_w = tf.Variable(tf.truncated_normal(shape = (400, 80), mean = mu, stddev = sigma), name = 'fl2_w') 
  fl2_b = tf.Variable(tf.zeros(80), name = 'fl2_b')
  fl3 = tf.matmul(fl2, fl2_w) + fl2_b
  #activation function relu
  fl3 = tf.nn.relu(fl3)
  #Dropout
  fl3_drop = tf.nn.dropout(fl3, dropout_kept_prob)
  #print(fl3)
    
  '''sixth fully connected layer'''
  fl3_w = tf.Variable(tf.truncated_normal(shape = (80, 10), mean = mu, stddev = sigma), name = 'fl3_w')
  fl3_b = tf.Variable(tf.zeros(10), name = 'fl3_b')
  flf = tf.matmul(fl3, fl3_w) + fl3_b
  

  
    
    
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