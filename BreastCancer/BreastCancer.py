# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# column_names = ['Sample code number', 'Clump Thickness', 'Cell Size',
#                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
#                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
#                    names = column_names)
#
#
# data = data.replace(to_replace='?', value=np.nan)   #将？替换为np.nan
# data = data.dropna(how='any')   #丢弃带有缺失值的数据！
#
# x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
#                                                      test_size=0.25, random_state=33)

train = pd.read_csv('cancer_train.csv')
test = pd.read_csv('cancer_test.csv')

def exchange(value):
    if value == 2:
        values = 0
    elif value == 4:
        values = 1
    return values

# 分割特征与分类目标
x_train = np.float32(train[['Clump Thickness', 'Uniformity of Cell Size']].T)
y_train = np.float32(train['Class'].T)

# for i in range(y_train.shape[0]):
#     y_train[i] = exchange(y_train[i])
# print(y_train[:5])

x_test = np.float32(test[['Clump Thickness', 'Uniformity of Cell Size']].T)
y_test = np.float32(test['Class'].T)

#定义一个tf变量b作为线性模型的截距，初始值为1.0
b = tf.Variable(tf.zeros(1))
#定义一个tf变量W作为线性模型的系数，初始值为-1.0到1.0之间均匀分布的随机数
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

#显式定义这个线性函数
y = tf.matmul(W, x_train) +b

#使用tf中的reduce_mean取得训练集上均方误差
loss = tf.reduce_mean(tf.square(y-y_train))

optimizer = tf.train.GradientDescentOptimizer(0.01)

#以最小二乘损失为优化目标
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(0,1000):
        sess.run(train)
        if step %200 == 0:
            print(step, sess.run(W), sess.run(b))

    test_negative = test.loc[test['Class'] ==2][['Clump Thickness', 'Uniformity of Cell Size']]
    test_postive = test.loc[test['Class'] ==4][['Clump Thickness', 'Uniformity of Cell Size']]



    plt.scatter(test_negative['Clump Thickness'], test_negative['Uniformity of Cell Size'],\
                marker='o', s=200, c='red')
    plt.scatter(test_postive['Clump Thickness'], test_postive['Uniformity of Cell Size'],\
                marker='x', s=150, c='black')

    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')

    lx = np.arange(0,12)

    #以3为分界面，计算方式如下
    ly = (3 - sess.run(b) -lx *sess.run(W)[0][0]) / sess.run(W)[0][1]

    plt.plot(lx, ly, color = 'green')
    plt.show()
