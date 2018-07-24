# 图像渐变和边缘检测

import cv2
import numpy as np

img = cv2.imread('../data/cluo.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# canny 边缘检测
edges = cv2.Canny(img, 100, 200)
cv2.imshow('Edges', edges)

cv2.imshow('Original', img)
cv2.imshow('Res', res)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()