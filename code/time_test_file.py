import tensorflow as tf
import timeit


W = tf.Variable(tf.random_normal((4800, 10000)))
x = tf.Variable(tf.random_normal((1000, 4800)))
y = tf.matmul(x, W)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start = timeit.default_timer()
_y = sess.run(y)
stop = timeit.default_timer()
print('Time taken: {}'.format(stop - start))
