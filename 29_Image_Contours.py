# Contours
# Contours can be explained simply as a curve joining all the continuous points
# (along the boundary) having the same color or intensity.
#
# Contours are a useful tool for shape analysis and object detection.
#
# For better accuracy:
# - Use binary images
# - Apply edge detection or thresholding before finding contours
#
# findContours function manipulates the original image,
# so always pass a copy if needed.
#
# findContours works best when:
# - Object is white
# - Background is black
#
# We have to find and draw contours as per the requirement.

import cv2
import numpy as np

# ----------------------------------------
# Load image
# ----------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\logo.jpg")

# Convert to grayscale
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
# threshold(img, thresh_value, max_value, type)
ret, thresh = cv2.threshold(img1, 20, 255, 0)

# ----------------------------------------
# Find Contours
# findContours(image, contour_retrieval_mode, method)
# ----------------------------------------
cnts, hier = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Total contours:", len(cnts))

# ----------------------------------------
# Draw Contours
# drawContours(image, contours, contour_id, color, thickness)
# If contour_id = -1 → draw all contours
# ----------------------------------------
img_contour = cv2.drawContours(img.copy(), cnts, -1, (25, 100, 15), 4)

# ----------------------------------------
# Display Results
# ----------------------------------------
cv2.imshow("original", img)
cv2.imshow("gray", img1)
cv2.imshow("thresh", thresh)
cv2.imshow("contours", img_contour)

cv2.waitKey(0)
cv2.destroyAllWindows()
