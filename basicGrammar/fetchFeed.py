# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:13:21 2017

@author: xqy12
"""

import tensorflow as tf
 
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    # 取回多个tensor
    result = sess.run([mul, intermed])
    print(result)


#feed的使用
#提供feed数据作为RUN()调用的参数

#tf.placeholder创建占位符
input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4, input5)

with tf.Session() as sess:
    
    print(sess.run([output], feed_dict={input4:[7.], input5:[2.0]}))
    