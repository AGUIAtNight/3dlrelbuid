# 读取所有棋盘格图像
import cv2
import glob
import os
import numpy as np

imgpoints_left, imgpoints_right = [], []  # 存储图像中的角点
objpoints = []  # 存储模板中的角点
images = glob.glob('./right/*.jpg')  # 所有棋盘格图像所在的目录
for fname in images:
    l = cv2.imread(fname)
    img_left.append(l)

images = glob.glob('./left/*.jpg')  # 所有棋盘格图像所在的目录
for fname in images:
    r = cv2.imread(fname)

    img_right.append(r)

print("获取角点", "left")
get_corners(img_left, corners_left)
print("获取角点", "right")
get_corners(img_right, corners_right)
