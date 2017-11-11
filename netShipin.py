from ops import *
import timeit
import time
#from ops import batch_norm, testing, evaluate
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten

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
  mu = 0
  sigma = 0.1
  
  '''first convolutional layer'''
  conv1_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 3, 80), mean = mu, stddev = sigma), name = 'conv1_w')
  conv1_w = tf.get_variable("conv1_w", shape = [5,5,3,80], initializer=tf.contrib.layers.xavier_initializer())
  conv1_b = tf.Variable(tf.zeros(80), name = 'conv1_b')
  conv1 = tf.nn.conv2d(input, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv1_b #(28,28,80)
  #activation function relu
  conv1 = tf.nn.relu(conv1)
  #norm1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'norm1')
  #max pooling
  conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')#(14,14,64)
  
  '''second convolutional layer'''
  convv_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 80, 160), mean = mu, stddev = sigma), name = 'convv_w')
  convv_b = tf.Variable(tf.zeros(160), name = 'convv_b')
  convv = tf.nn.conv2d(conv1, convv_w, strides = [1,1,1,1], padding = 'VALID') + convv_b #(10,10,160)  
  convv = batch_norm(convv, is_training, name = "bn_convv")
  convv = tf.nn.relu(convv) #(10,10,160)
  #print("shape of convv: ", convv)
  
  '''third convolutional layer'''
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
  
  '''forth fully connected layer'''
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
  #print(flf)
  

  
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
  
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  y = tf.placeholder(tf.int32, (None))
  #one_hot_y = tf.one_hot(y,10)
  
  '''data preparation'''
  cifar10_train = Cifar10(batch_size = 100, one_hot = True, test = False, shuffle = True)
  cifar10_test = Cifar10(batch_size = 100, one_hot = False, test = True, shuffle = False)
  test_images, test_labels = cifar10_test.images, cifar10_test.labels
  
  
  lr = 0.003
  logits = net(x, True, 0.3)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y)
  loss = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate = lr)
  grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)
  #training_operation = optimizer.minimize(grads_and_vars)
  #train = tf.train.AdamOptimizer.minimize(loss)
  
  '''create summary'''
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/histogram", var)

  add_gradient_summaries(grads_and_vars)
  tf.summary.scalar('loss_operation', loss)
  merged_summary_op= tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('logs/')
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
  accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  saver = tf.train.Saver(max_to_keep = 5)
  sess = tf.Session()
  
  try:
    print("================Trying to restore last checkpoint ...================")
    saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
    print("================Checkpoint restored .================")
  except:
    print("================Failed to find last check point================")
    sess.run(tf.global_variables_initializer())
  
  
  '''training'''
  start_time = time.time()
  with sess:
    
    print("Training")
    global_step = 0
    for i in range(20):
      for offset in range(0, 500):
        batch_x, batch_y = cifar10_train.get_next_batch()
        _, summaries = sess.run([training_operation, merged_summary_op], feed_dict = {x: batch_x, y: batch_y})
        if global_step % 100 == 1:
          _loss = sess.run(loss, feed_dict = {x:batch_x, y:batch_y})
          print("steps: ", global_step, "|time consume: %.2f" % ((time.time() - start_time)/3600), "hours. || Current Loss: ", _loss)
          summary_writer.add_summary(summaries, global_step = global_step)
        global_step += 1
      ''' 
      #validation_accuracy = evaluate(test_images, test_labels, accuracy_operation)
      total_accuracy = 0
      test_x, test_y = cifar10_test.get_next_batch()
      accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
      total_accuracy += (accuracy * len(batch_x))     
      '''
      training_accuracy = evaluate(batch_x, batch_y, accuracy_operation, x, y)
      print("sets: ", i)
      print("accuracy: ", training_accuracy * 100, " % ")
      print()
      saver.save(sess, 'ckpt/netCheckpoint', global_step = i)
  


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
  
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  
  logits = net(x, False, 1.0)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
    test_accuracy = testing(cifar10_test_images, logits, x)
    #print(test_accuracy)
    output = np.argmax(test_accuracy, 1)
  return output
