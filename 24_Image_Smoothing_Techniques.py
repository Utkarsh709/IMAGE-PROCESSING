# There are so many filters which are used for smoothing the image.
# There are Low Pass Filters (LPF) which are used to remove noise from images.
# There are High Pass Filters which are used to detect and find edges in an image.

# We discuss various filters like:
# homogeneous, blur (averaging), gaussian, median, bilateral

import cv2
import numpy as np

# Read the image
img = cv2.imread(r"D:\Image_Processing and Computer Vision\noisy.jpg")
img = cv2.resize(img, (400, 400))
cv2.imshow("original", img)

# ----------------------------------------
# FILTER NUMBER 1 : Homogeneous Filter
# ----------------------------------------
# This filter works such that each output pixel is the mean of its kernel neighbours
# It is also known as homogeneous filter where all pixels contribute equally
# Kernel is a small matrix applied to the image
# Formula: (1 / kernel_width * kernel_height) * kernel

kernel = np.ones((5, 5), np.float32) / 25
h_filter = cv2.filter2D(img, -1, kernel)   # -1 keeps same image depth
cv2.imshow("homogeneous", h_filter)

# ----------------------------------------
# FILTER NUMBER 2 : Averaging / Blur Filter
# ----------------------------------------
# Takes the average of all pixels under kernel area
# Replaces the central pixel with this average value

blur = cv2.blur(img, (5, 5))   # image and kernel size
cv2.imshow("blur", blur)

# ----------------------------------------
# FILTER NUMBER 3 : Gaussian Filter
# ----------------------------------------
# Uses weighted kernel
# Center pixels have higher weight than surrounding pixels
# It reduces noise while preserving overall structure

gau = cv2.GaussianBlur(img, (5, 5), 0)   # 0 means sigma is auto-calculated
cv2.imshow("gaussian blur", gau)

# ----------------------------------------
# FILTER NUMBER 4 : Median Filter
# ----------------------------------------
# Computes median of all pixels under kernel window
# Central pixel is replaced with median value
# Highly effective for removing salt-and-pepper noise
# Kernel size must be odd

med = cv2.medianBlur(img, 5)
cv2.imshow("median filter", med)

# ----------------------------------------
# FILTER NUMBER 5 : Bilateral Filter
# ----------------------------------------
# Highly effective at noise removal while preserving edges
# Works like Gaussian filter but considers pixel intensity differences
# It is slower compared to other filters

# arguments: (image, diameter, sigma_color, sigma_space)
bi_f = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("bilateral filter", bi_f)

# ----------------------------------------
# Wait and destroy windows
# ----------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
