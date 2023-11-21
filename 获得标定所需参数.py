import cv2
import glob
import numpy as np

'''
获得标定所需参数
'''
# 定义棋盘格的大小
chessboard_size = (8, 5)
# 定义图像分辨率，根据自己相机的分辨率修改
imgsz = (1280, 720)
# 定义棋盘格中每个格子的物理大小，自己用尺子量，单位为毫米（mm）
square_size = 6

# 定义棋盘格模板的点的坐标
# 生成每个角点三维坐标，共有chessboard_size[0]*chessboard_size[1]个坐标，z轴置0不影响
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]
                       ].T.reshape(-1, 2) * square_size  # 计算得到每个角点的x,y

# 读取所有棋盘格图像并提取角点
imgpoints_left, imgpoints_right = [], []  # 存储图像中的角点
objpoints = []  # 存储模板中的角点
images = glob.glob('./right/*.jpg')  # 所有棋盘格图像所在的目录
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_size, None)  # 计算corner
        ret, corners = cv2.find4QuadCornerSubpix(
            gray, corners, (7, 7))  # 提高角点检测的准确性和稳定性
        if ret == True:
            imgpoints_right.append(corners)
            objpoints.append(objp)
    except Exception as e:
        print("发生异常:", e)
    # ret, corners = cv2.findChessboardCorners(
    #       gray, chessboard_size, None)  # 计算corner
    # ret, corners = cv2.find4QuadCornerSubpix(
    #       gray, corners, (7, 7))  # 提高角点检测的准确性和稳定性
    # if ret == True:
    #       imgpoints_right.append(corners)
    #       objpoints.append(objp)

images = glob.glob('./left/*.jpg')  # 所有棋盘格图像所在的目录
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, chessboard_size, None)  # 计算corner
    ret, corners = cv2.find4QuadCornerSubpix(
        gray, corners, (7, 7))  # 提高角点检测的准确性和稳定性
    if ret == True:
        imgpoints_left.append(corners)
'''
开始标定，获得参数
'''
# 标定相机，获得内参和畸变参数
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray.shape[::-1], None, None)
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray.shape[::-1], None, None)

# 指定迭代次数最大30或者误差小于0.001
term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 进行双目相机标定,主要是获得R，T两个矩阵
rotation_matrix, translation_matrix = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l,
    mtx_r, dist_r,
    imgsz, flags=cv2.CALIB_FIX_INTRINSIC, criteria=term)[5:7]

# 获得矫正矩阵和投影矩阵，用于后续进行图像校正
rect_left, rect_right, \
    proj_left, proj_right, \
    dispartity, \
    ROI_left, ROI_right = cv2.stereoRectify(
        mtx_l, dist_l,
        mtx_r, dist_r,
        imgsz, rotation_matrix, translation_matrix,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)


'''
打印结果
'''
print('mtx_l = np.array({})'.format(np.array2string(mtx_l, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('mtx_r = np.array({})'.format(np.array2string(mtx_r, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('dist_l = np.array({})'.format(np.array2string(dist_l, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('dist_r = np.array({})'.format(np.array2string(dist_r, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('R = np.array({})'.format(np.array2string(rotation_matrix, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('T = np.array({})'.format(np.array2string(translation_matrix, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('rect_left = np.array({})'.format(np.array2string(rect_left, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('rect_right = np.array({})'.format(np.array2string(rect_right,
      separator=', ', formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('proj_left = np.array({})'.format(np.array2string(proj_left, separator=', ',
      formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('proj_right = np.array({})'.format(np.array2string(proj_right,
      separator=', ', formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
print('dispartity = np.array({})'.format(np.array2string(dispartity,
      separator=', ', formatter={'int': lambda x: f'{x: 3d}'}, prefix='[', suffix=']')))
# print('mtx_l = np.array({})'.format(mtx_l))
# print('mtx_r = np.array({})'.format(mtx_r))
# print('dist_l = np.array({})'.format(dist_l))
# print('dist_r = np.array({})'.format(dist_r))
# print('R = np.array({})'.format(rotation_matrix))
# print('T = np.array({})'.format(translation_matrix))
# print('rect_left = np.array({})'.format(rect_left))
# print('rect_right = np.array({})'.format(rect_right))
# print('proj_left = np.array({})'.format(proj_left))
# print('proj_right = np.array({})'.format(proj_right))
# print('dispartity = np.array({})'.format(dispartity))
print('ROI_left = np.array({})'.format(ROI_left))
print('ROI_right = np.array({})'.format(ROI_right))
