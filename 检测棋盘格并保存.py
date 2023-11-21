import cv2
import pysnooper


import os
current_dir = os.getcwd()

# 创建左文件夹
left_folder = "left"
left_folder_path = os.path.join(current_dir, left_folder)
os.makedirs(left_folder_path, exist_ok=True)

right_folder = "right"
right_folder_path = os.path.join(current_dir, right_folder)
os.makedirs(right_folder_path, exist_ok=True)


# @pysnooper.snoop()
def detect_chessboard_and_save(frame, save_path):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 定义棋盘格的大小
    chessboard_size = (8, 5)
    # 查找棋盘格角点
    # ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    left_img = frame[:, 0:640, :]
    right_img = frame[:, 640:1280, :]
    gray1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    try:
        ret, corners = cv2.findChessboardCorners(
            gray1, chessboard_size, None)  # 计算corner
        ret1, corners = cv2.find4QuadCornerSubpix(
            gray1, corners, (7, 7))  # 提高角点检测的准确性和稳定性
        ret, corners = cv2.findChessboardCorners(
            gray2, chessboard_size, None)  # 计算corner
        ret2, corners = cv2.find4QuadCornerSubpix(
            gray2, corners, (7, 7))  # 提高角点检测的准确性和稳定性

        # ret1, corner = cv2.findChessboardCorners(left_img, (8, 5))
        # ret2, corner = cv2.findChessboardCorners(right_img, (8, 5))
        if ret1 and ret2:
            global i
            i += 1
            # 在图像上绘制角点
            # cv2.drawChessboardCorners(frame, board_size, corners, ret)

            # 拍摄棋盘格并保存

            left_img = frame[:, 0:640, :]
            right_img = frame[:, 640:1280, :]

            cv2.imwrite(left_folder_path + '\\'+'{}.jpg'.format(str(save_path)),
                        left_img)  # 保存左图
            cv2.imwrite(right_folder_path + '\\'+'{}.jpg'.format(str(save_path)),
                        right_img)  # 保存右图

            # cv2.imwrite('{save_path}.jpg', frame)
            print("棋盘格图像保存成功！")
    except Exception as e:
        print("请调整相机位置，确保棋盘格位于图像中央！", e)


# 摄像头捕获
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
i = 0
while cap.isOpened():

    # 读取摄像头图像帧
    ret, frame = cap.read()

    # 检测棋盘格并保存
    if ret:
        cv2.imwrite("{save_path.jpg", frame)
        detect_chessboard_and_save(frame, i)

    # 显示图像
    # cv2.imshow('frame', frame)

    # 检测键盘输入，如果按下 q 键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()

# 关闭所有图像窗口
cv2.destroyAllWindows()
