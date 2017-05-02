# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:45:51 2017

@author: xqy12
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data

minist = input_data.read_data_sets("MNIST_data/", one_hot=True)