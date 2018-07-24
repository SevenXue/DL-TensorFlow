# 图像的腐蚀和膨胀
# 腐蚀有利于去除一些白噪声

import cv2
import numpy as np

img = cv2.imread('../data/cluo.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 280])

msk = cv2.inRange(hsv, low_red, upper_red)
res = cv2.bitwise_and(img, img, mask=msk)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(msk, kernel, iterations=1)
dilation = cv2.dilate(msk, kernel, iterations=1)

cv2.imshow('Original', img)
cv2.imshow('Mask', msk)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)

k = cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()