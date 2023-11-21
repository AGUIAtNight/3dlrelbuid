import cv2
import os
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 生成目录
path = './calibration/'
path_l = './calibration/left/'
path_r = './calibration/right/'
os.mkdir(path) if not os.path.exists(path) else None
os.mkdir(path_l) if not os.path.exists(path_l) else None
os.mkdir(path_r) if not os.path.exists(path_r) else None
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('img', frame)
    k = cv2.waitKey(1)
    # 按下ESC退出
    if k == 27:
        break
    # 按下空格键暂停
    if k == 32:
        cv2.imshow('img', frame)
        # 再次按下空格保存
        if cv2.waitKey() == 32:
            cv2.imwrite(path + "{}.jpg".format(count), frame)  # 保存全图
            cv2.imwrite(path_l + "{}.jpg".format(count),
                        frame[:, 0:640])  # 保存左图
            cv2.imwrite(path_r + "{}.jpg".format(count),
                        frame[:, 640:])  # 保存右图
            count += 1
cv2.destroyAllWindows()
cap.release()
