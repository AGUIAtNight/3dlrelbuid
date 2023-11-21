import cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while (1):
    _, frame = cap.read()
    assert _, "摄像头获取失败"
    cv2.imshow('img', frame)
    c = cv2.waitKey(1)
    if c == 27:
        cap.release()
        break
