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
        
    index = 0
    correct_count = 0
    score = 0
    count_both_correct = 0
    count_one_correct = 0
    none_correct = 0
    for item in batch_y:
        c = []
        for n in item:
            c.append(n)
        if c == out[index]:
            correct_count += 1
            score += 3
            count_both_correct += 1
        elif ((c[0] == out[index][0]) or (c[1] == out[index][1])):
            score += 1
            count_one_correct += 1
        else:
            none_correct += 1
        #print(c, out[index])
        index += 1 
    print("Both digits correct: " + str(count_both_correct))
    print("One digit correct: " + str(count_one_correct))
    print("Both digits incorrect: " + str(none_correct))
    print("Score: " + str(score) + " out of " + str(len(batch_y)*3))
    return out, correct_count/len(out)*100    
    