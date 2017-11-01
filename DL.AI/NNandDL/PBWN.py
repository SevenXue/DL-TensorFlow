import numpy as np

def sigmoid_derivative(x):

    s = 1/(1+np.exp(-x))
    ds = s*(1-s)

    return ds

def softmax(x):
    x_exp = np.exp(x)

    x_sum = x_exp.sum(axis=1).reshape(x.shape[0],1)

    s = x_exp/x_sum

    return s


