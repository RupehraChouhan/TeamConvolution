# Reference: https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
import tensorflow as tf
import numpy as np
BATCH_SIZE = 100

# TODO: Optionally create utility functions for convolution, fully connected layers here

def batch_norm(input, is_training, momentum=0.9, epsilon=1e-5, in_place_update=True, name="batch_norm"):
  # Example batch_norm usage:
  # bn0 = batch_norm(conv1, is_training=is_training, name='bn0')
  if in_place_update:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=name)
  else:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)


def evaluate(X_data, y_data, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples  
  
  
def testing(X_data, score, x):
    print("Testing accuracy")
    sess = tf.get_default_session()
    batch_x = X_data
    logist = sess.run(score, feed_dict={x: batch_x})
    
    return logist
  
  
def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)