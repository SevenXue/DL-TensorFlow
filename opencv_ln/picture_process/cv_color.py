import cv2
import numpy as np

img = cv2.imread('../data/mainlogo.png')

# hsv颜色提取
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('img', img)
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 图像模糊和平滑
# 均值模糊
kernel = np.ones((126, 126), np.float32) / 15876
smoothed = cv2.filter2D(res, -1, kernel)
cv2.imshow('Original', img)
cv2.imshow('Averaging', smoothed)

# 高斯模糊
blur = cv2.GaussianBlur(res, (126, 126), 0)
cv2.imshow('Gaussian Blurring', blur)

# 中值模糊
median = cv2.medianBlur(res, 126)
cv2.imshow('Median Blur', median)

k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

