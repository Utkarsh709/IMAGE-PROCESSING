import numpy as np
import cv2 as cv

"""
Feature Detection and Description
---------------------------------
Corner Detection using Harris Corner Detector

Corners are points in the image where intensity changes
in both x and y directions. Harris detector finds such points
using image gradients.
"""

# -------------------------------
# Read image
# -------------------------------
img = cv.imread(r"D:\Image_Processing and Computer Vision\shapes.png")

if img is None:
    print("Error: Image not found")
    exit()

cv.imshow("Original Image", img)

# -------------------------------
# Convert to grayscale
# Harris requires grayscale + float32
# -------------------------------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

# -------------------------------
# Harris Corner Detection
# Parameters:
# blockSize = 2  -> neighborhood size
# ksize     = 3  -> Sobel aperture
# k         = 0.04 -> Harris free parameter
# -------------------------------
res = cv.cornerHarris(gray, 2, 3, 0.04)

# -------------------------------
# Dilate result for better visibility
# -------------------------------
res = cv.dilate(res, None)

# -------------------------------
# Threshold and mark corners
# -------------------------------
img[res > 0.01 * res.max()] = [0, 0, 255]   # Red color

# -------------------------------
# Display result
# -------------------------------
cv.imshow("Harris Corners", img)

cv.waitKey(0)
cv.destroyAllWindows()
