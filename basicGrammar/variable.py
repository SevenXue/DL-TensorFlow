"""
Created on Mon Apr 17 20:58:28 2017

@author: xqy12
"""
import tensorflow as tf

# 创建一个变量， 初始化为标量0
state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后， 变量必须先经过’初始化‘（init）op初始化
# 首先必须增加一个’初始化‘op到图中
init_op =tf.fglobal_variables_initializers()

with tf.Session() as sess:
    # 运行 ’init‘ op
    #将所有的variables进行初始化
    sess.run(init_op)
    print(sess.run(state))
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
