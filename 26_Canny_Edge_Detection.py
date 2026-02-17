# Canny Edge Detection using OpenCV
# Canny Edge Detection is a popular edge detection approach.
# It uses a multi-stage algorithm to detect edges.
# It was developed by John F. Canny in 1986.
# This approach combines 5 steps:
# 1) Noise reduction (Gaussian)
# 2) Gradient calculation
# 3) Non-maximum suppression
# 4) Double threshold
# 5) Edge tracking by hysteresis

import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread(r"D:\Image_Processing and Computer Vision\strom_breaker.JPG")
img = cv2.resize(img, (600, 600))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny(img, threshold1, threshold2)
# threshold1 -> lower threshold
# threshold2 -> higher threshold
canny = cv2.Canny(img_gray, 50, 150)

cv2.imshow("original", img)
cv2.imshow("gray", img_gray)
cv2.imshow("canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
