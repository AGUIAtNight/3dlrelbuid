
import cv2
import os


import cv2
import numpy as np

import cv2

# 加载图片
image = cv2.imread('input.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 400, 100)
# kernel = np.ones((2, 2), np.uint8)

# dilation = cv2.dilate(edges, kernel, iterations=1)

# 输出结果
cv2.imwrite('output.jpg', edges)
