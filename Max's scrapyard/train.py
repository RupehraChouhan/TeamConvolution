import tensorflow as tf
import numpy as np
from loadData import TextureImages
from tensorflow.contrib.layers import flatten
from utility import *
import math

def model(x, dropout, is_training):
    keep_prob = 1-dropout
    #conv 1
    #input: (64, 64, 1)
    #output: (30, 30, 16)
    conv1_w = tf.get_variable(shape = [5,5,1,16], name = 'conv1_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv1_b = tf.Variable(tf.zeros(16), name = 'conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID')
    h_conv1 = tf.nn.relu(conv1 + conv1_b)
    pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # print(pool1)
    
    #conv 2
    #input: (30, 30, 16)
    #output: (13, 13, 32)
    conv2_w = tf.get_variable(shape = [5,5,16,32], name = 'conv2_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv2_b = tf.Variable(tf.zeros(32), name = 'conv2_b')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1, 1, 1, 1], padding = 'VALID')
    bn_conv2 = batch_norm(conv2, is_training, name = "bn_conv2")
    h_conv2 = tf.nn.relu(bn_conv2 + conv2_b)
    pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # print(pool2)  
    
    #conv 3
    #input: (13, 13, 32)
    #output: (4, 4, 64)
    conv3_w = tf.get_variable(shape = [5,5,32,64], name = 'conv3_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv3_b = tf.Variable(tf.zeros(64), name = 'conv3_b')
    conv3 = tf.nn.conv2d(pool2, conv3_w, strides = [1, 1, 1, 1], padding = 'VALID')
    bn_conv3 = batch_norm(conv3, is_training, name = "bn_conv3")
    h_conv3 = tf.nn.relu(bn_conv3 + conv3_b)
    pool3 = tf.nn.max_pool(h_conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    # print(pool3)      
    
    #fully connected 1
    #input: (4, 4, 64) --> (1024)
    #output: (256)
    fc_w = tf.get_variable(shape = [1024, 256], name = "fc_w", initializer = tf.contrib.layers.xavier_initializer())
    fc_b = tf.Variable(tf.zeros(256), name = 'fc_b')
    h_dropout = tf.nn.dropout(pool3, keep_prob)
    # print(h_dropout)
    reshape = flatten(h_dropout)
    h_fc = tf.nn.relu(tf.matmul(reshape, fc_w) + fc_b)
    # print(h_fc)
    
    #fully connected 2
    #input: (256)
    #output: (64)
    fc2_w = tf.get_variable(shape = [256, 64], name = "fc2_w", initializer = tf.contrib.layers.xavier_initializer())
    fc2_b = tf.Variable(tf.zeros(64), name = 'fc2_b')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc, fc2_w) + fc2_b)
    # print(h_fc2)    
    
    
    #two softmax logit output
    #output: (10)
    s1_w = tf.get_variable(shape = [64, 10], name = "s1_w", initializer = tf.contrib.layers.xavier_initializer())
    s1_b = tf.Variable(tf.zeros(10), name = 's1_b')
    logit1 = tf.matmul(h_fc2, s1_w) + s1_b
    
    s2_w = tf.get_variable(shape = [64, 10], name = "s2_w", initializer = tf.contrib.layers.xavier_initializer())
    s2_b = tf.Variable(tf.zeros(10), name = 's2_b')
    logit2 = tf.matmul(h_fc2, s2_w) + s2_b    
    

    
    return logit1, logit2



def train(EPOCHS=900, BATCH_SIZE=64, drop_out=0.1, lr=0.003):

    
    train_set = TextureImages('train', batch_size = BATCH_SIZE)
    valid_set = TextureImages('valid', shuffle = False)
    
    #Uncomment this line for testing
    test_set = TextureImages('test', shuffle = False)
    
    tf.reset_default_graph()
    
    images, labels = train_set.get_next_batch()
    print("image shape:", images[0].shape)
    print("label shape:", labels[0].shape)
       
    x = tf.placeholder(tf.float32, (None, 64, 64, 1))
    y = tf.placeholder(tf.int32, (None,2))    
    is_training = tf.placeholder(tf.bool, ())
    
    logit1, logit2 = model(x, drop_out, is_training = True)
    
    
    global_step = tf.Variable(0)
    decaylr = tf.train.exponential_decay(lr, global_step, 5000, 0.096)
   
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit1, labels = y[:, 0])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit2, labels = y[:, 1]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss, global_step = global_step)
    
    
    steps  = BATCH_SIZE * EPOCHS
    saver = tf.train.Saver(max_to_keep = 3)
    sess = tf.Session()
    
    #use this function instead of accuracy when out of memory
    def evaluation(sets,BATCH_SIZE):
        validation_steps = math.ceil(len(sets.images) / BATCH_SIZE)
        valids1 = []
        valids2 = []
        validys = []
        for i in range(validation_steps):
            valid_x, valid_y = sets.get_next_batch()
            valid1, valid2 = sess.run([logit1, logit2], feed_dict = {x:valid_x})
    
            if (len(valids1) == 0):
                valids1 = valid1
            else:
                valids1 = np.concatenate((valids1, valid1), axis=0)
    
            if (len(valids2) == 0):
                valids2 = valid2
            else:
                valids2 = np.concatenate((valids2, valid2), axis=0)
    
            if (len(validys) == 0):
                validys = valid_y
            else:
                validys = np.concatenate((validys, valid_y), axis=0)
    
        acc = accuracy(valids1, valids2, validys)
        print("Accuracy: ", acc)        
        return acc  
    
    
    try:
        print("================Trying to restore last checkpoint ...================")
        saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
        print("================Checkpoint restored .================")
        print("================skip training process================")
        print("================testing================")
        print('Evaluating on test set')
        batch_x, batch_y = test_set.get_full_set()
        l1,l2 = sess.run([logit1, logit2], feed_dict = {x:batch_x})
        result, test_accuracy = accuracy(l1,l2,batch_y)
        #result is a list of list[digit1, digit2]
        #print("result", result)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        sess.close()    
        return test_accuracy
    except:
        print("================Failed to find last check point================")
        print("================initial global variables================")
        sess.run(tf.global_variables_initializer())    
    
   
    with sess:
        for i in range(steps):
            batch_x, batch_y = train_set.get_next_batch()
            _,l,l1,l2 = sess.run([optimizer, loss,logit1, logit2], feed_dict = {x:batch_x, y:batch_y})
            if i % 1000 == 1:
                print("step: ", i, "; Current loss: ", l)
                acc = accuracy(l1,l2,batch_y)
                print("Training accuracy: ", acc)
                saver.save(sess, 'ckpt/checkpoint', global_step = i)
            
        # Need to do a validation here after all the training

        acc = evaluation(valid_set,BATCH_SIZE)
        print("Validation accuracy: ", acc)

    return acc


if __name__ == "__main__":
    grid_search = False

    if (not grid_search):
        train()
    else:

        epochs = [400]

        batch_sizes = [64, 32, 16]
        dropout = [0.5, 0.8, 0.7, 0.6, 0.9, 0.4, 0.3, 0.2, 0.1]
        learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.01]

        print("Running Training with grid search " + str(len(epochs) * len(batch_sizes) * len(dropout) * len(learning_rates)) + " times.")

        my_file = open("hypers.csv", "w")
        my_file.write("epochs, batch_size, dropout, learning_rate, accuracy\n")
        my_file.close()

        for e in epochs:
            for b in batch_sizes:
                for d in dropout:
                    for lr in learning_rates:
                        print("Epochs: " + str(e) + "; Batch size: " + str(b) + "; Dropout: " + str(d) + "; Learning rate: " + str(lr))
                        my_file = open("hypers.csv", "a+")
                        my_file.write(str(e) + ", " + str(b) + ", " + str(d) + ", " + str(lr) + ", ")
                        my_file.close()
                        acc = train(EPOCHS=e, BATCH_SIZE=b, drop_out=d, lr=lr)
                        my_file = open("hypers.csv", "a+")
                        my_file.write(str(acc) + "\n")
                        my_file.close()

