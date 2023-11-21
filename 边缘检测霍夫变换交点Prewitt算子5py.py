import cv2
import numpy as np

# 读取图像
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 构建Prewitt算子模板
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# 分别对图像进行x轴和y轴方向的Prewitt边缘检测
outx = cv2.filter2D(img, cv2.CV_64F, kernelx)
outy = cv2.filter2D(img, cv2.CV_64F, kernely)

# 计算x轴和y轴方向梯度的绝对值
absx = cv2.convertScaleAbs(outx)
absy = cv2.convertScaleAbs(outy)

# 将x轴和y轴方向梯度的绝对值加权求和，得到最终的边缘图像
dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)


lines = cv2.HoughLines(dst, 1, np.pi/180, threshold=100)
# lines1 = cv2.HoughLines(edges, 1, np.pi/180, threshold=10)
# Find all intersection points
intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        rho1, theta1 = lines[i][0]
        rho2, theta2 = lines[j][0]
        A = [[np.cos(theta1), np.sin(theta1)], [
            np.cos(theta2), np.sin(theta2)]]
        b = [[rho1], [rho2]]
        try:
            x0, y0 = np.linalg.solve(A, b)
            intersections.append([int(x0), int(y0)])
        except:
            pass

# Count occurrences of each intersection point
counts = {}
for point in intersections:
    counts[tuple(point)] = counts.get(tuple(point), 0) + 1

# Extract the most common intersection point
most_common = max(counts, key=counts.get)


# vertical_lines = []
# for line in lines1:
#     rho, theta = line[0]
#     if (3.12) < theta < (3.16):
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         vertical_lines.append((x1, y1, x2, y2))


# Draw a circle around the most common intersection point
cv2.circle(img, tuple(most_common), 10, (0, 0, 255), 2)

# Find all lines that pass through the most common intersection point
lines_through_point = []
for line in lines:
    rho, theta = line[0]
    x0 = np.cos(theta) * rho
    y0 = np.sin(theta) * rho
    dx = most_common[0] - x0
    dy = most_common[1] - y0
    if abs(dx) < 20 and abs(dy) < 20:
        x1 = int(x0 - 1000 * (-dy))
        y1 = int(y0 - 1000 * dx)
        x2 = int(x0 + 1000 * (-dy))
        y2 = int(y0 + 1000 * dx)
        lines_through_point.append((x1, y1, x2, y2))

# Draw all lines that pass through the most common intersection point
for line in lines_through_point:
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1)

# Draw all lines that pass through the most common intersection point
# 可视化垂直线
# for line in vertical_lines:
#     x1, y1, x2, y2 = line
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# 显示结果图像
# cv2.imshow('result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Save final image
# cv2.imwrite("output.jpg", img)


# end = time.time()

# print('Running time: %s Seconds' % (end-start))


cv2.imwrite("output点Prewitt算子5.jpg", dst)
