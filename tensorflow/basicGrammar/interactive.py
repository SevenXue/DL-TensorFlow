# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 20:47:29 2017

@author: xqy12
"""

import tensorflow as tf
sess = tf.InteractiveSession ()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的run()方法初始化'x'
x.initializer.run()

# 增加一个减法sub
sub = tf.subtract(x, a)
print(sub.eval())