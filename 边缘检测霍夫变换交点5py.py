import cv2
import numpy as np

# 读取图像
img = cv2.imread("input.jpg")

# Convert to grayscale and apply Canny edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 进行边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 进行水平方向和垂直方向的概率霍夫线变换
# 进行概率霍夫线变换
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, maxLineGap=0)

# 计算每条直线的角度
angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi  # 计算角度
    angles.append(angle)

# 找到水平线和垂直线的角度范围（可以根据实际情况进行微调）
h_angle_range = (2, -2)
v_angle_range = (88, 92)

# 分别提取水平线和垂直线
h_lines = [lines[i] for i in range(len(angles)) if abs(angles[i]) < abs(
    h_angle_range[0]) and abs(angles[i]) > h_angle_range[1]]
v_lines = [lines[i] for i in range(len(angles)) if abs(angles[i]) > abs(
    v_angle_range[0]) and abs(angles[i]) < abs(v_angle_range[1])]

# 绘制水平线
for line in h_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

# 绘制垂直线
for line in v_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)


cv2.imwrite("output5.jpg", img)
