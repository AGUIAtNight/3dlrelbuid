# 视差计算
import cv2
import numpy as np


def Computegradientofpixelintensityimage(trueDisp_left, left_image):
    disparity_map = trueDisp_left  # 视差图

    # 计算视差图的梯度
    disparity_gradient_x = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    disparity_gradient_y = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)

    intensity_map = left_image  # 像素强度图

    # 转换为灰度图像
    gray_intensity = cv2.cvtColor(intensity_map, cv2.COLOR_BGR2GRAY)

    # Computegradientofpixelintensityimage
    intensity_gradient_x = cv2.Sobel(gray_intensity, cv2.CV_64F, 1, 0, ksize=3)
    intensity_gradient_y = cv2.Sobel(gray_intensity, cv2.CV_64F, 0, 1, ksize=3)

    # 计算视差梯度和像素强度梯度的绝对值
    disparity_gradient = np.sqrt(
        disparity_gradient_x**2 + disparity_gradient_y**2)
    intensity_gradient = np.sqrt(
        intensity_gradient_x**2 + intensity_gradient_y**2)

    # 计算可靠度值，并进行归一化处理
    reliability_map = disparity_gradient / (intensity_gradient + 1e-6)
    reliability_map = cv2.normalize(
        reliability_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    threshold = 150  # 设置阈值
    reliable_pixels = (reliability_map >= threshold).astype(
        np.uint8) * 255  # 高于阈值的设为可靠，低于阈值的设为不可靠
    print(reliable_pixels)


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 4 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(
            left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(
            right_image_down, left_image_down)
        disparity_left = cv2.resize(
            disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(
            disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    # import cv2
    # from ximgproc import *
    # import cv2
    # from cv2.ximgproc import fastGlobalSmootherFilter
    # from FastGlobalSmoothFilter算法 import fast_global_smoother_filter

    # imgSmoothl = fast_global_smoother_filter(
    #     trueDisp_left, 625.0)
    # imgSmoothr = fast_global_smoother_filter(
    #     trueDisp_right, 625.0)
    # 计算具有可靠视差的视差图

    # Computegradientofpixelintensityimage(trueDisp_left, left_image)
    # Computegradientofpixelintensityimage(trueDisp_right, right_image)
    # import 深度图
    # 深度图.depth_map(trueDisp_left, trueDisp_right)
    # import 点云
    # 点云.disparity_to_point_cloud(trueDisp_left, trueDisp_right)

    return trueDisp_left, trueDisp_right
