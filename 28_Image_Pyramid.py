"""
We use Image Pyramid because sometimes we work on the same image
but at different resolutions.
For example: face detection, eye detection, object searching etc.

In such cases, we create a set of images with different resolutions.
This set of images is called an Image Pyramid.

We also use image pyramids for image blending.
"""

# There are two types of Image Pyramid:
# 1) Gaussian Pyramid
# 2) Laplacian Pyramid

import cv2
import numpy as np

# ----------------------------------------
# Load image
# ----------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img = cv2.resize(img, (700, 700))

# ----------------------------------------
# Gaussian Pyramid
# Gaussian Pyramid has two functions:
# 1) cv2.pyrDown() -> reduces image size
# 2) cv2.pyrUp()   -> increases image size
# ----------------------------------------

# ----------- PYR DOWN -----------
pd1 = cv2.pyrDown(img)   # first downsample
pd2 = cv2.pyrDown(pd1)   # second downsample

# ----------- PYR UP -------------
pu1 = cv2.pyrUp(pd2)     # upsample
pu2 = cv2.pyrUp(pu1)     # upsample again

# ----------------------------------------
# Display results
# ----------------------------------------
cv2.imshow("original", img)
cv2.imshow("pd1", pd1)
cv2.imshow("pd2", pd2)
cv2.imshow("pu1", pu1)
cv2.imshow("pu2", pu2)

cv2.waitKey(0)
cv2.destroyAllWindows()
