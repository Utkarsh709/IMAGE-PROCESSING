# -------------------------------------------------------------------
# ---------------- MORPHOLOGICAL OPENING & CLOSING -------------------
# -------------------------------------------------------------------
# Morphological Opening and Closing are ADVANCED morphological
# transformations built using erosion and dilation.
#
# These operations are mainly used for:
# - Noise removal
# - Shape correction
# - Filling gaps or removing small objects
#
# They are normally applied on BINARY IMAGES.
# -------------------------------------------------------------------

import cv2
import numpy as np

# -------------------------------------------------------------------
# READ IMAGE IN GRAYSCALE
# -------------------------------------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\col_balls.jpg", 0)

# -------------------------------------------------------------------
# CONVERT IMAGE TO BINARY
# Using THRESH_BINARY_INV so foreground becomes WHITE
# -------------------------------------------------------------------
_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# -------------------------------------------------------------------
# ---------------------- OPENING ------------------------------------
# -------------------------------------------------------------------
# Opening:
# - Opening is just another name for EROSION followed by DILATION
#
# Order:
# 1) Erosion
# 2) Dilation
#
# Effect:
# - Removes small white noises
# - Breaks thin connections
# - Smoothens object boundaries
#
# Common use:
# - Noise removal (salt noise)
# -------------------------------------------------------------------
kernel_open = np.ones((4, 4), np.uint8)  # 4x4 kernel

opening = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel_open
    # iterations parameter can be added if required
)

# -------------------------------------------------------------------
# ---------------------- CLOSING ------------------------------------
# -------------------------------------------------------------------
# Closing:
# - Closing is just another name for DILATION followed by EROSION
#
# Order:
# 1) Dilation
# 2) Erosion
#
# Effect:
# - Fills small holes inside objects
# - Closes small black gaps
# - Connects nearby white regions
#
# Common use:
# - Filling gaps in objects
# -------------------------------------------------------------------
kernel_close = np.ones((3, 3), np.uint8)  # 3x3 kernel

closing = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel_close
)

# -------------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------------
cv2.imshow("Original Image", img)
cv2.imshow("Binary Mask", mask)
cv2.imshow("Kernel (Opening)", kernel_open)
cv2.imshow("Opening Result", opening)
cv2.imshow("Closing Result", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
