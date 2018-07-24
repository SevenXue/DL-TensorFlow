# open 开放的目标是消除“假阳性”，例如背景中的像素噪声
# close 关闭的想法是消除假阴性。

import cv2
import numpy as np

img = cv2.imread('../data/cluo.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
# res = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

res = cv2.bitwise_and(img, img, mask=opening)

cv2.imshow('Original', img)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('res', res)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()