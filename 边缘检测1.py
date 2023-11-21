import cv2
import numpy as np

# 加载图片
image = cv2.imread('input.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 进行膨胀操作，增加线条粗细
kernel = np.ones((50, 50), np.uint8)
dilated = cv2.dilate(edges, kernel)

# 寻找轮廓
contours, hierarchy = cv2.findContours(
    dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
for i in range(len(contours)):
    cnt = contours[i]

    # 计算轮廓的面积和周长
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # 如果轮廓不符合条件，则跳过处理
    if area < 50 or perimeter < 50:
        continue

    # 进一步处理交叉的线条
    approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
    if len(approx) < 3:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    roi = image[y:y + h, x:x + w].copy()
    cv2.imwrite('output/' + str(i) + '.jpg', roi)
