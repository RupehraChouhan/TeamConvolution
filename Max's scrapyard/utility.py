import tensorflow as tf

BATCH_SIZE = 64
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
    #for offset in range(0, num_examples, BATCH_SIZE):
        #batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
    accuracy = sess.run(accuracy_operation, feed_dict={x: X_data, y: y_data})
    total_accuracy += (accuracy * len(X_data))
    return total_accuracy / num_examples