import numpy as np
from scipy.ndimage.filters import gaussian_filter


def fast_global_smoother_filter(image, lambda_val, sigma=20):
    # 将图像转换为浮点类型，并将像素值缩放到 [0, 1] 范围内
    image = image.astype(np.float32) / 255.0

    # 计算灰度图像
    gray_image = np.mean(image, axis=2)

    # 计算梯度图像
    grad_x = np.gradient(gray_image)[1]
    grad_y = np.gradient(gray_image)[0]

    # 计算全局系数
    global_coeff = 1.0 / (1.0 + lambda_val * (np.abs(grad_x) + np.abs(grad_y)))

    # 分别对各个通道进行平滑处理
    smoothed_r = gaussian_filter(image[:, :, 0], sigma)
    smoothed_g = gaussian_filter(image[:, :, 1], sigma)
    smoothed_b = gaussian_filter(image[:, :, 2], sigma)

    # 合并通道并使用全局系数进行平滑
    smoothed_image = np.dstack((smoothed_r, smoothed_g, smoothed_b))
    smoothed_image = global_coeff[:, :, np.newaxis] * image + \
        (1 - global_coeff[:, :, np.newaxis]) * smoothed_image

    # 将像素值重新缩放回 [0, 255] 范围内，并将图像类型转换为无符号整型
    smoothed_image = (smoothed_image * 255).astype(np.uint8)

    return smoothed_image


def fast_global_smoother_filter1(image, lambda_val, sigma):
    # 将图像转换为浮点类型，并将像素值缩放到 [0, 1] 范围内
    gray_image = image.astype(np.float32) / 255.0

    # 不再计算灰度图像，因为输入已经是灰度图像

    # 计算梯度图像
    grad_x = np.gradient(gray_image)[1]
    grad_y = np.gradient(gray_image)[0]

    # 计算全局系数
    global_coeff = 1.0 / (1.0 + lambda_val * (np.abs(grad_x) + np.abs(grad_y)))

    # 对灰度图像进行平滑处理
    smoothed_gray = gaussian_filter(gray_image, sigma)

    # 使用全局系数进行平滑
    smoothed_image = global_coeff[:, :, np.newaxis] * gray_image + \
        (1 - global_coeff[:, :, np.newaxis]) * smoothed_gray

    # 将像素值重新缩放回 [0, 255] 范围内，并将图像类型转换为无符号整型
    smoothed_image = (smoothed_image * 255).astype(np.uint8)

    return smoothed_image


def fast_global_smoother_filter2(image, lambda_val):
    # 将图像转换为浮点类型，并将像素值缩放到 [0, 1] 范围内
    gray_image = image.astype(np.float32) / 255.0

    # 计算梯度图像
    grad_x = np.gradient(gray_image)[1]
    grad_y = np.gradient(gray_image)[0]

    # 计算全局系数
    global_coeff = 1.0 / (1.0 + lambda_val * (np.abs(grad_x) + np.abs(grad_y)))

    # 使用全局系数进行平滑
    smoothed_image = global_coeff[:, :] * gray_image

    # 将像素值重新缩放回 [0, 255] 范围内，并将图像类型转换为无符号整型
    smoothed_image = (smoothed_image * 255).astype(np.uint8)

    return smoothed_image
