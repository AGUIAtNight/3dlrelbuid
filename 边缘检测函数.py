import cv2
import numpy as np


def fill_contours(img):
    # 读取图像
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 定义Prewitt算子
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # 应用Prewitt算子进行边缘检测
    edges_x = cv2.filter2D(img, -1, kernel_x)
    edges_y = cv2.filter2D(img, -1, kernel_y)

    # 组合x和y方向的边缘检测结果
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    # 保存边缘检测结果图像
    # cv2.imwrite(output_path, edges)
    return edges
