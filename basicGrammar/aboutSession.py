

import tensorflow as tf
hello = tf.constant('hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
sess.close()

# import tensorflow as tf
#
# # 创建原op，不需要任何输入
# matrix1 = tf.constant([[3., 3.]])
# matrix2 = tf.constant([[2.],[2.]])
#
# product = tf.matmul(matrix1, matrix2)
#
# #启动默认图
# #启动图的第一步是创建一个Session对象
# #sess = tf.Session()
#
# # 调用sess的run()方法来执行矩阵乘法
# # run()方法相当于一个触发器，使得语句开始执行，且并发执行
#
# '''
# result = sess.run(product)
# print(result)
# sess.close()
# '''
#
# # 使用with语句的优点是执行完可以自动关闭
# with tf.Session() as sess:
#     result = sess.run([product])
#     print(result)

