# -------------------------------------------------------------------
# ---------------- MORPHOLOGICAL TRANSFORMATIONS ---------------------
# -------------------------------------------------------------------
# Morphological transformations are simple image processing operations
# based on the SHAPE of objects in an image.
#
# These operations are normally performed on BINARY IMAGES.
#
# It requires TWO inputs:
# 1) Original image (binary image)
# 2) Structuring element (kernel)
#
# The two BASIC morphological operations are:
# 1) Erosion
# 2) Dilation
#
# Kernel:
# - A small matrix (mostly filled with 1s)
# - Slides over the image pixel by pixel
# - Decides how pixels are modified
# -------------------------------------------------------------------

import cv2
import numpy as np

# -------------------------------------------------------------------
# READ IMAGE IN GRAYSCALE
# Morphological operations work better on binary images
# -------------------------------------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\col_balls.jpg", 0)

# -------------------------------------------------------------------
# CONVERT IMAGE INTO BINARY
# THRESH_BINARY_INV is used so object becomes WHITE (255)
# and background becomes BLACK (0)
# -------------------------------------------------------------------
_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# -------------------------------------------------------------------
# CREATE KERNEL (STRUCTURING ELEMENT)
# 5x5 kernel filled with ones
# -------------------------------------------------------------------
kernel = np.ones((5, 5), np.uint8)

# -------------------------------------------------------------------
# ---------------------- EROSION ------------------------------------
# -------------------------------------------------------------------
# Erosion:
# - Erodes (shrinks) the boundaries of foreground objects
# - A pixel in the output is WHITE only if ALL pixels under
#   the kernel are WHITE
#
# Effect:
# - Removes small white noises
# - Shrinks foreground objects
# - Breaks thin connections
# -------------------------------------------------------------------
erosion = cv2.erode(mask, kernel, iterations=1)

# -------------------------------------------------------------------
# ---------------------- DILATION -----------------------------------
# -------------------------------------------------------------------
# Dilation:
# - Opposite of erosion
# - A pixel in the output is WHITE if AT LEAST ONE pixel
#   under the kernel is WHITE
#
# Effect:
# - Increases white region
# - Enlarges foreground objects
# - Fills small holes
#
# Note:
# - Often erosion is followed by dilation to remove noise
#   without losing object structure
# -------------------------------------------------------------------
kernel2 = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(mask, kernel2, iterations=1)

# -------------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------------
cv2.imshow("Original Image", img)
cv2.imshow("Binary Mask", mask)
cv2.imshow("Kernel", kernel)
cv2.imshow("Erosion Result", erosion)
cv2.imshow("Dilation Result", dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
