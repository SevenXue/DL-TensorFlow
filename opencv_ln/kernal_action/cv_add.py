import cv2
import numpy as np
import matplotlib.pyplot as plt

#加法运算

img1 = cv2.imread('../data/3D-matplotlib.png')
img2 = cv2.imread('../data/mainsvmimage.png')

add = img1 + img2

add1 = cv2.add(img1, img2)

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

cv2.imshow('weighted', weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

