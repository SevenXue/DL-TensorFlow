# 图片的阈值处理

import cv2
import numpy as np

img = cv2.imread('../data/bookpage.jpg')

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#使用阈值对图像进行处理
retval, threshold = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

# 自动化学习阈值
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

cv2.imshow('original', grayscaled)
cv2.imshow('threshold', th)
cv2.waitKey(0)
cv2.destroyAllWindows()