import cv2
img = cv2.imread('uTools_1686814505814.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corner = cv2.findChessboardCorners(img1, (8, 5))
ret, corner = cv2.find4QuadCornerSubpix(img1, corner, (7, 7))
cv2.drawChessboardCorners(img, (9, 6), corner, ret)
cv2.imshow('corner', img)
cv2.waitKey(0)
