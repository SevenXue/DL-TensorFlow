import numpy as np

# A = np.array([[56.0, 0.0, 4.4, 68.0],
#               [1.2, 104.0, 52.0, 8.0],
#               [1.8, 135.0, 99.0, 0.9]])
#
# print(A)
# cal = A.sum(axis=0)
# print(cal)
# percentage = 100*A/cal.reshape(1,4)
# print(percentage)

#notes
# a = np.random.randn(5,1)
# ## not use!!!
# a = np.random.randn(5)

a1 = np.random.randn(3, 3)
print(a1)
b1 = np.random.randn(3,1)
print(b1)
c = a1*b1
print(c)
print(np.sum(c))

import math
