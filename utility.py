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
  
  
def accuracy(l1,l2,batch_y):
  #hard coded accuracy 
    out = []
    index = 0
    for item in l1:
        holder = []
        temp = []
        for num in item:
            temp.append(num)
        holder.append(temp.index(max(item)))
        out.append(holder)
    for item in l2:
        temp = []
        for num in item:
            temp.append(num)
        out[index].append(temp.index(max(item)))
        out[index].sort()
        index = index + 1
    lab = [] 
    id = 0
    correct_count = 0
    for item in batch_y:
        c = []
        for n in item:
            c.append(n)
        #lab.append(a)
        if c == out[id]:
            correct_count += 1
        #print(c, out[id])
        id += 1  
    return correct_count/len(out)*100