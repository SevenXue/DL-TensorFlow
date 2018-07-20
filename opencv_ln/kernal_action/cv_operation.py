import cv2
import numpy as np
import matplotlib.pyplot as plt

#按位运算

img1 = cv2.imread('../data/cl.jpg')
img2 = cv2.imread('../data/opencv_logo.jpg')

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

dst = cv2.add(img1_bg, img2_fg)

img1[0:rows, 0:cols] = dst

plt.subplot(231), plt.imshow(img2gray)
plt.subplot(232), plt.imshow(mask)
plt.subplot(233), plt.imshow(dst)
plt.subplot(234), plt.imshow(img1_bg)
plt.subplot(235), plt.imshow(img2_fg)
plt.subplot(236), plt.imshow(img1)

plt.show()

