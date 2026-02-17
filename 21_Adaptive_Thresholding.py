# ---------------------------------------------
# THRESHOLDING IN OPENCV
# ---------------------------------------------
# Simple Thresholding:
# - Uses ONE fixed threshold value for the entire image
# - Works well only when lighting conditions are uniform
# - Fails when image has shadows, highlights, or uneven illumination
#
# Adaptive Thresholding:
# - Calculates threshold value for SMALL REGIONS of the image
# - Different threshold for different areas
# - Very useful when lighting conditions vary across the image
# - Commonly used in document scanning, number plate detection,
#   medical images, and real-world camera inputs
#
# Even though simple thresholding exists, adaptive thresholding
# is required because real-world images rarely have uniform lighting.
# ---------------------------------------------

import cv2
import numpy as np

# Read image in GRAYSCALE
# Adaptive thresholding works ONLY on single-channel (grayscale) images
img = cv2.imread(r"D:\Image_Processing and Computer Vision\page.jpg", 0)

# Resize image for better visualization
img = cv2.resize(img, (400, 400))

# ---------------------------------------------
# SIMPLE THRESHOLDING
# ---------------------------------------------
# Uses a fixed threshold value (127)
# All pixels > 127 -> WHITE (255)
# All pixels <= 127 -> BLACK (0)
_, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# ---------------------------------------------
# ADAPTIVE MEAN THRESHOLDING
# ---------------------------------------------
# Threshold value = mean of neighborhood pixels - C
# Block size = 11 (size of local region, must be odd)
# C = 2 (constant subtracted from mean)
th2 = cv2.adaptiveThreshold(
    img,
    255,                            # maximum value
    cv2.ADAPTIVE_THRESH_MEAN_C,     # adaptive mean method
    cv2.THRESH_BINARY,              # binary threshold
    11,                             # neighborhood size
    2                               # constant
)

# ---------------------------------------------
# ADAPTIVE GAUSSIAN THRESHOLDING
# ---------------------------------------------
# Threshold value = weighted sum (Gaussian) of neighborhood pixels - C
# Gives better results when noise and lighting variation are present
th3 = cv2.adaptiveThreshold(
    img,
    255,                                # maximum value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # adaptive gaussian method
    cv2.THRESH_BINARY,                  # binary threshold
    11,                                 # neighborhood size
    2                                   # constant
)

# ---------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------
cv2.imshow("Original Image", img)
cv2.imshow("Simple Threshold", th1)
cv2.imshow("Adaptive Mean Threshold", th2)
cv2.imshow("Adaptive Gaussian Threshold", th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
