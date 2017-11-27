from ops import *
import timeit
from cifar10 import Cifar10
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import os



def add_gradient_summaries(grads_and_vars):
  for grad, var in grads_and_vars:
    if grad is not None:
      tf.summary.histogram(var.op.name + "/gradient", grad)

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
  # raise NotImplementedError
  # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,3,16], stddev=0.1))
  b_conv1 = tf.Variable(tf.zeros(16))
  conv_layer1 = tf.nn.conv2d(input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
  conv_layer1 = conv_layer1 + b_conv1
  conv_layer1_bn = batch_norm(conv_layer1, is_training=is_training, name='conv_layer1_bn')
  conv_layer1 = tf.nn.relu(conv_layer1_bn) # Activation

  # Layer 2: Convolutional. Input = 32x32x3. Output = 28x28x6.
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5,5,16,8], stddev=0.1))
  b_conv2 = tf.Variable(tf.zeros(8))
  conv_layer2 = tf.nn.conv2d(conv_layer1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
  conv_layer2 = conv_layer2 + b_conv2
  conv_layer2_bn = batch_norm(conv_layer2, is_training=is_training, name='conv_layer2_bn') #+ input # 1x1 conv with stride
  conv_layer2 = tf.nn.relu(conv_layer2_bn)  # Activation
  conv_layer_flat2 = tf.contrib.layers.flatten(conv_layer2) # Flatten

  # Layer 3: Convolutional. Input = 32x32x3. Output = 28x28x6.
  W_conv3 = tf.Variable(tf.truncated_normal(shape=[5,5,8,4], stddev=0.1))
  b_conv3 = tf.Variable(tf.zeros(4))
  conv_layer3 = tf.nn.conv2d(conv_layer2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
  conv_layer3 = conv_layer3 + b_conv3
  conv_layer3_bn = batch_norm(conv_layer3, is_training=is_training, name='conv_layer3_bn') #+ input # 1x1 conv with stride
  conv_layer3 = tf.nn.relu(conv_layer3_bn)  # Activation
  #print(conv_layer3.shape, conv_layer1.shape)
  #conv_layer3 = conv_layer3 + conv_layer1 #skip layer
  conv_layer_pool3 = tf.nn.max_pool(conv_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  conv_layer_flat3 = tf.contrib.layers.flatten(conv_layer_pool3) # Flatten

  # Layer 3: Fully Connected. Input
  W_fc4 = tf.Variable(tf.truncated_normal(shape=[1024,256], stddev=0.1))
  b_fc4 = tf.Variable(tf.zeros(256))
  fc4 = tf.nn.relu(tf.matmul(conv_layer_flat3, W_fc4) + b_fc4) # Activation
  fc4_drop = tf.nn.dropout(fc4, dropout_kept_prob)
  
  #Layer 5:
  W_fc5 = tf.Variable(tf.truncated_normal(shape=[256,64], stddev=0.1))
  b_fc5 = tf.Variable(tf.zeros(64))
  fc5 = tf.nn.relu(tf.matmul(fc4_drop, W_fc5) + b_fc5) # Activation
  
  # Use dropout
  #fc5_drop = tf.nn.dropout(fc5, dropout_kept_prob)

  # Layer 6: Fully Connected logits. Input
  W_fc6 = W_conv1 = tf.Variable(tf.truncated_normal(shape=[64,10], stddev=0.1))
  b_fc6 = tf.Variable(tf.zeros(10))
  logits = tf.nn.relu(tf.matmul(fc5, W_fc6) + b_fc6)  # Activation

  return logits


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
  cifar_train = Cifar10(test=False, shuffle=False, one_hot=True)
  cifar10_test = Cifar10(batch_size=100, one_hot=False, test=True, shuffle=False)  
  X_train = cifar_train._images
  y_train = cifar_train._labels  
  EPOCHS = 15
  BATCH_SIZE = 128
  dropoutprob = tf.placeholder(tf.float32)
  training = True
  x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
  y = tf.placeholder(tf.int32, (None,10)) 
  #one_hot_y = tf.one_hot(y, 10)
  rate = 0.0005
  #dropoutprob = 0
  logits = net(x,training,dropoutprob)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
  loss_operation = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate = rate)
  grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)   
  add_gradient_summaries(grads_and_vars)
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
      batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
      accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropoutprob:1})
      total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples  
    
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      num_examples = len(X_train)

      print("Training...")
      print()
      for i in range(EPOCHS):
          X_train, y_train = shuffle(X_train, y_train)
          for offset in range(0, num_examples, BATCH_SIZE):
              end = offset + BATCH_SIZE
              batch_x, batch_y = X_train[offset:end], y_train[offset:end]
              #print("Print batch_x...")
              #print(batch_x)
              #print("Print batch_y...")
              #print(batch_y)
              sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropoutprob: 0.5})
          validation_accuracy = evaluate(X_train, y_train)
          print("EPOCH {} ...".format(i + 1))
          print("Validation Accuracy = {:.3f}".format(validation_accuracy))
          
      
      try:
          saver
      except NameError:
          saver = tf.train.Saver()
      save_path = 'ci10model'
      if not os.path.isabs(save_path):
          save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
      saver.save(sess, save_path)
      print("Model saved")
    
    
def test(cifar10_test_images):
  print("In TEST function")
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # Get test set
  cifar10_train = Cifar10(test=True, shuffle=False, one_hot=True)
  x_test, y_test = cifar10_train._images, cifar10_train._labels
  cifar10_train_images = tf.cast(x_test, tf.float32)
  number_of_images = cifar10_train_images.shape[0]
  
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  y = tf.placeholder(tf.int32, (None,10))
  training = False
  dropoutprob = tf.placeholder(tf.float32)
  EPOCHS = 10
  BATCH_SIZE = 128
  logits = net(x,training,dropoutprob)
  training_yp = tf.argmax(logits,1)
  Y = []
  saver = tf.train.Saver()
  save_path = 'ci10model'
  if not os.path.isabs(save_path):
      save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))  
  with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, save_path)
    prediction_labels = []
    predictions = sess.run(logits, feed_dict={x: cifar10_test_images, dropoutprob: 1.0})
    print("Prediction: ", predictions[0])
    for array in predictions:
        predicted_class = 0
        max_score = array[0]

        for i, j in enumerate(array):
            if j > max_score:
                max_score = j
                predicted_class = i

        prediction_labels.append([predicted_class])

    print(prediction_labels)
    return np.array(prediction_labels)
    