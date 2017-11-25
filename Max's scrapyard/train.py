import tensorflow as tf
import numpy as np
from loadData import TextureImages
from tensorflow.contrib.layers import flatten
from utility import *

def model(x, keep_prob, is_training):
    #conv 1
    #input: (64, 64, 1)
    #output: (30, 30, 16)
    conv1_w = tf.get_variable(shape = [5,5,1,16], name = 'conv1_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv1_b = tf.Variable(tf.zeros(16), name = 'conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID')
    h_conv1 = tf.nn.relu(conv1 + conv1_b)
    pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    print(pool1)
    
    #conv 2
    #input: (30, 30, 16)
    #output: (13, 13, 32)
    conv2_w = tf.get_variable(shape = [5,5,16,32], name = 'conv2_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv2_b = tf.Variable(tf.zeros(32), name = 'conv2_b')
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1, 1, 1, 1], padding = 'VALID')
    h_conv2 = tf.nn.relu(conv2 + conv2_b)
    pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    print(pool2)  
    
    #conv 3
    #input: (13, 13, 32)
    #output: (4, 4, 64)
    conv3_w = tf.get_variable(shape = [5,5,32,64], name = 'conv3_w', initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv3_b = tf.Variable(tf.zeros(64), name = 'conv3_b')
    conv3 = tf.nn.conv2d(pool2, conv3_w, strides = [1, 1, 1, 1], padding = 'VALID')
    h_conv3 = tf.nn.relu(conv3 + conv3_b)
    pool3 = tf.nn.max_pool(h_conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    print(pool3)      
    
    #fully connected 1
    #input: (4, 4, 64) --> (1024)
    #output: (256)
    fc_w = tf.get_variable(shape = [1024, 256], name = "fc_w", initializer = tf.contrib.layers.xavier_initializer())
    fc_b = tf.Variable(tf.zeros(256), name = 'fc_b')
    h_dropout = tf.nn.dropout(pool3, keep_prob)
    print(h_dropout)
    reshape = flatten(h_dropout)
    h_fc = tf.nn.relu(tf.matmul(reshape, fc_w) + fc_b)
    print(h_fc)
    
    #fully connected 2
    #input: (256)
    #output: (64)
    fc2_w = tf.get_variable(shape = [256, 64], name = "fc2_w", initializer = tf.contrib.layers.xavier_initializer())
    fc2_b = tf.Variable(tf.zeros(64), name = 'fc2_b')
    h_fc2 = tf.nn.relu(tf.matmul(h_fc, fc2_w) + fc2_b)
    print(h_fc2)    
    
    
    #two softmax logit output
    s1_w = tf.get_variable(shape = [64, 10], name = "s1_w", initializer = tf.contrib.layers.xavier_initializer())
    s1_b = tf.Variable(tf.zeros(10), name = 's1_b')
    logit1 = tf.matmul(h_fc2, s1_w) + s1_b
    
    s2_w = tf.get_variable(shape = [64, 10], name = "s2_w", initializer = tf.contrib.layers.xavier_initializer())
    s2_b = tf.Variable(tf.zeros(10), name = 's2_b')
    logit2 = tf.matmul(h_fc2, s2_w) + s2_b    
    

    
    return logit1, logit2


def train():
    
    EPOCHS = 10
    BATCH_SIZE = 64
    drop_out = 0.5
    NUM_ITERS  = int(2000 / BATCH_SIZE * EPOCHS)
    
    train_set = TextureImages('train', batch_size=BATCH_SIZE)
    test_set = TextureImages('test', shuffle=False)
    
    tf.reset_default_graph()
    
    images, labels = train_set.get_next_batch()
    print("iamge shape:", images[0].shape)
    print("label shape:", labels[0].shape)
       
    x = tf.placeholder(tf.float32, (None, 64, 64, 1))
    y = tf.placeholder(tf.int32, (None,2))    
    is_training = tf.placeholder(tf.bool, ())
    
    logit1, logit2 = model(x, drop_out, is_training = is_training)
    
    ##reshape the flattened vector into a image
    ##a = images[0].reshape(64,64)
    #f = open('out.txt', 'w')
    #for image in images:
        #for item in image:
            #temp = np.array2string(item).replace('\n','')
            #f.write(temp)
            #f.write('\n')
    #f.close
    #print(images[0].shape)
    global_step = tf.Variable(0)
    lr = 0.001
    decaylr = tf.train.exponential_decay(lr, global_step, 10000, 0.096)
    #print(logit1, logit2)
    #print(y)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit1, labels = y[:, 0])) + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit2, labels = y[:, 1]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss, global_step = global_step)
    
    #if tf.greater_equal(tf.argmax(logit1, 1), tf.argmax(logit2, 1)):
    prediction = tf.logical_and(tf.equal(tf.argmax(logit2, 1), tf.argmax(y[0])), tf.equal(tf.argmax(logit1, 1), tf.argmax(y[1])))
    #else: 
    #prediction = tf.equal(tf.argmax(logit1, 1), tf.argmax(y, 1)) and tf.equal(tf.argmax(logit2, 1), tf.argmax(y, 2))
        
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    steps = 5000
    sess = tf.Session()
    with sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch_x, batch_y = train_set.get_next_batch()
            _,l,l1,l2 = sess.run([optimizer, loss,logit1, logit2], feed_dict = {x:batch_x, y:batch_y})
            if i % 1000 == 1:
                print("step: ", i, " |Current loss: ", l)
                training_accuracy = evaluate(batch_x, batch_y, accuracy, x, y)
                print("Training accuracy: ", training_accuracy)
                
                #print(l1)
                #for item in l1:
                    #temp=[]
                    #for num in item:
                        #temp.append(num)
                    ##print(item)
                    #print(temp.index(max(item)))
                #print(y[:, 0])

def main():
    train()
if __name__ == "__main__":
    main()