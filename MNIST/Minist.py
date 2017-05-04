
"""
Created on Thu Apr 20 08:58:03 2017

@author: xqy12
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data

minist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#占位符，None代表张量的第一个维度可以是任何长度
x = tf.placeholder(tf.float32, [None, 784])

#W为权重，b代表偏置量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# softmax
y = tf.nn.softmax(tf.matmul(x,W) + b)


y_ = tf.placeholder(tf.float32, [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = minist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: minist.test.images, y_: minist.test.labels}))



