import cv2
import numpy as np
from matplotlib import pyplot as plt

# use cv2.imread(), cv2.imshow(), cv2.imwrite()


img = cv2.imread('../data/cluo.jpg', 0)
# b, g, r = cv2.split(img)
# img2 = cv2.merge((r, g, b))

cv2.imshow('bgr', img)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('../data/cluogray.png', img)
    cv2.destroyAllWindows()
