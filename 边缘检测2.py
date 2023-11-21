import cv2
import numpy as np

# Load image
img = cv2.imread('input.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on output image
output = np.zeros_like(img)
for cnt in contours:
    # Approximate the contour with a polygon
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # If the polygon has more than 2 vertices, draw it as a line
    if len(approx) > 1:
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

# Save output image
cv2.imwrite('output.png', output)
