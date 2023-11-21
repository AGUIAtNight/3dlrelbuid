# import 视差计算
import cv2
import glob
import os
import numpy as np


def get_corners(imgs, corners):
    for img in imgs:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 9x12棋盘有8x11个角点
            ret, c = cv2.findChessboardCorners(gray, (8, 5))
            assert (ret)
            ret, c = cv2.find4QuadCornerSubpix(gray, c, (7, 7))
            assert (ret)
            corners.append(c)
        except AssertionError:
            print("assertion error")
            continue


mtx_l = np.array([[990.9992544,   0., 255.77496849],
                  [0., 982.03121621, 257.33982351],
                  [0.,   0.,   1.]])
mtx_r = np.array([[996.86283867,   0., 372.66824011],
                  [0., 994.45882779, 203.59233342],
                  [0.,   0.,   1.]])
dist_l = np.array([[-6.61759052e-01,  1.46735038e+00,  3.06624602e-03,  1.11889872e-03,
                    -3.19970246e+00]])
dist_r = np.array(
    [[-0.42459846,  0.43149149, -0.00584208, -0.00560596, -0.62122897]])
R = np.array([[0.99686024, -0.00255278, -0.07913997],
              [0.0065506,  0.9987128,  0.05029739],
              [0.07890971, -0.05065789,  0.99559381]])
T = np.array([[-56.8560311],
              [0.76523124],
              [1.07410098]])
rect_left = np.array([[0.99501371, -0.01503351, -0.09859876],
                      [0.01750031,  0.99955391,  0.02420154],
                      [0.09819094, -0.02580637,  0.99483294]])
rect_right = np.array([[0.99973109, -0.01345548, -0.01888651],
                       [0.01297841,  0.9995992, -0.02515944],
                       [0.01921748,  0.02490756,  0.99950503]])
proj_left = np.array([[988.245022,   0., 489.65180206,   0.],
                      [0., 988.245022, 122.49853134,   0.],
                      [0.,   0.,   1.,   0.]])
proj_right = np.array([[9.88245022e+02,  0.00000000e+00,  4.89651802e+02, -5.62028033e+04],  [0.00000000e+00,  9.88245022e+02,  1.22498531e+02,  0.00000000e+00],
                       [0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00]])
dispartity = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.89651802e+02],  [0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.22498531e+02],
                       [0.00000000e+00,  0.00000000e+00,
                           0.00000000e+00,  9.88245022e+02],
                       [0.00000000e+00,  0.00000000e+00,  1.75835539e-02, -0.00000000e+00]])
ROI_left = np.array((117, 0, 1163, 590))
ROI_right = np.array((79, 0, 1201, 606))
img_left = []
img_right = []
corners_left = []
corners_right = []
img_file = glob.glob('./*.jpg')
imgsize = (640, 480)

# for img in img_file:
#     try:

#         frame = cv2.imread(img)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         l = frame[:, 0:640]
#         r = frame[:, 640:]
#         if l .shape == (480, 640) and r.shape == (480, 640):
#             img_left.append(l)
#             img_right.append(r)
#         # print("获取角点", "left")
#         # get_corners(img_left, corners_left)
#         # print("获取角点", "right")
#         # get_corners(img_right, corners_right)
#     except Exception as e:
#         print(e)
# print("获取角点", "left")
# get_corners(img_left, corners_left)
# print("获取角点", "right")
# get_corners(img_right, corners_right)


def get_corners1(img, corners):

    # 读取所有棋盘格图像并提取角点
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


def Correction(l, r):
    # for i in range(len(img_left)):
    # l = img_left[i]
    # r = img_right[i]
    # 计算双目校正的矩阵
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, imgsize, R, T)
    # 计算校正后的映射关系
    maplx, maply = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, imgsize, cv2.CV_16SC2)
    maprx, mapry = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, imgsize, cv2.CV_16SC2)
    # 映射新图像
    lr = cv2.remap(l, maplx, maply, cv2.INTER_LINEAR)
    rr = cv2.remap(r, maprx, mapry, cv2.INTER_LINEAR)
    all = np.hstack((lr, rr))
    cv2.imwrite("saved_image.jpg", all)
    # 变换之后和变换之前的角点坐标不一致，所以线不是正好经过角点，只是粗略估计，但偶尔能碰到离角点比较近的线，观察会比较明显
    # cv2.line(all, (-1, int(corners_left[i][0][0][1])),
    #          (all.shape[1], int(corners_left[i][0][0][1])), (255), 1)
    # 可以看出左右图像y坐标对齐还是比较完美的，可以尝试着打印双目校正前的图片，很明显，左右y坐标是不对齐的
    # cv2.imshow('a', all)
    # c = cv2.waitKey()
    # cv2.destroyAllWindows()
    # if c == 27:
    #     break

    # trueDisp_left, trueDisp_right = 视差计算.stereoMatchSGBM(lr, rr)
    import need.ointCloudfromimages as ointCloudfromimages
    ointCloudfromimages.create_point_cloud(lr,
                                           rr)

    # all = np.hstack((trueDisp_left, trueDisp_right))
    # cv2.imwrite('all.jpg',
    #             all)  # 保存左图

    # cv2.imshow('a', all)
    # import time
    # time.sleep(0.1)
    # # c = cv2.waitKey()
    # # cv2.destroyAllWindows()
    # output_path = 'output_video.mp4'  # 视频保存路径和文件名
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    # fps = 30  # 每秒帧数
    # frame_size = (1280, 480)  # 视频帧大小
    # video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


# print("end")
