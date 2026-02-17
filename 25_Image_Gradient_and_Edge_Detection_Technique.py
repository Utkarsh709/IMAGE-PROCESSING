# Image Gradient
# It is a directional change in the color or intensity in an image.
# It is the most important part to find information from an image.
# Like finding edges within the image.
# There are various methods to find image gradient.
# These are: Laplacian Derivatives, SobelX and SobelY.
# All these functions have different mathematical approaches to get results.
# All load image in grayscale.

import cv2
import numpy as np

# ----------------------------------------
# Load image and convert to grayscale
# ----------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\captain_america.jpg")
img = cv2.resize(img, (400, 300))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("original", img)
cv2.imshow("gray", img_gray)

# ----------------------------------------
# Laplacian Gradient
# ----------------------------------------
# parameter(img, data_type for -ve values, ksize)
# Laplacian detects edges in all directions

lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

# ----------------------------------------
# Sobel Operation
# ----------------------------------------
# Sobel is a joint Gaussian smoothing + differentiation operation
# So it is more resistant to noise
# Used for x and y direction gradients

# parameter(img, type for -ve values, x, y, ksize)
# SobelX focuses on vertical edges
# SobelY focuses on horizontal edges

sobelX = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # x direction
sobelY = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # y direction

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# ----------------------------------------
# Combine SobelX and SobelY
# ----------------------------------------
sobelcombine = cv2.bitwise_or(sobelX, sobelY)

# ----------------------------------------
# Display Results
# ----------------------------------------
cv2.imshow("Laplacian", lap)
cv2.imshow("SobelX", sobelX)
cv2.imshow("SobelY", sobelY)
cv2.imshow("Combined image", sobelcombine)

cv2.waitKey(0)
cv2.destroyAllWindows()
