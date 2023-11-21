import cv2
import numpy as np

# Load image
img = cv2.imread('output1.jpg')

# Convert to grayscale and binarize
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill contours on black mask image
mask = np.zeros_like(img)
for contour in contours:
    cv2.fillPoly(mask, [contour], (255, 255, 255))

# Merge the mask and the original image
output = cv2.addWeighted(img, 0.7, mask, 0.3, 0)

# Save output image
cv2.imwrite('output.png', output)
