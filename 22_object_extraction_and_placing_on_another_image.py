# ------------------------------------------------------------
# OBJECT EXTRACTION AND PLACING ON ANOTHER IMAGE
# (ROI / Background Subtraction)
# ------------------------------------------------------------
# Goal:
# - Extract foreground object from image2
# - Remove its background using masking
# - Place that object into image1 at a selected ROI
#
# This technique is widely used in:
# - Logo insertion
# - Image compositing
# - Augmented reality
# - Background replacement
# ------------------------------------------------------------

import cv2
import numpy as np

# ------------------------------------------------------------
# LOAD TWO IMAGES
# img1 : Background image (where object will be placed)
# img2 : Foreground image (object to be extracted)
# ------------------------------------------------------------
img1 = cv2.imread(r"D:\Image_Processing and Computer Vision\hero1.jpg")
img2 = cv2.imread(r"D:\Image_Processing and Computer Vision\strom_breaker.JPG")

# Resize images for compatibility
img1 = cv2.resize(img1, (1024, 650))
img2 = cv2.resize(img2, (600, 650))

# ------------------------------------------------------------
# SELECT REGION OF INTEREST (ROI) FROM img1
# ROI size must be SAME as img2 size
# ------------------------------------------------------------
r, c, ch = img2.shape
print("Image2 shape:", r, c, ch)

# Extract top-left ROI from img1
roi = img1[0:r, 0:c]

# ------------------------------------------------------------
# CONVERT img2 TO GRAYSCALE
# Mask creation works better on grayscale images
# ------------------------------------------------------------
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# CREATE BINARY MASK USING THRESHOLDING
# - Object becomes WHITE
# - Background becomes BLACK
# ------------------------------------------------------------
_, mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

# ------------------------------------------------------------
# INVERT MASK
# - Background becomes WHITE
# - Object becomes BLACK
# Used to remove object area from ROI
# ------------------------------------------------------------
mask_inv = cv2.bitwise_not(mask)

# ------------------------------------------------------------
# REMOVE OBJECT AREA FROM ROI (BACKGROUND PART)
# Only background region of img1 is kept
# ------------------------------------------------------------
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# ------------------------------------------------------------
# EXTRACT ONLY OBJECT FROM img2 (FOREGROUND PART)
# ------------------------------------------------------------
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# ------------------------------------------------------------
# COMBINE BACKGROUND AND FOREGROUND
# ------------------------------------------------------------
res = cv2.add(img1_bg, img2_fg)

# ------------------------------------------------------------
# PLACE RESULT BACK INTO ORIGINAL IMAGE
# ------------------------------------------------------------
final = img1.copy()
final[0:r, 0:c] = res

# ------------------------------------------------------------
# DISPLAY ALL STEPS (FOR LEARNING PURPOSE)
# ------------------------------------------------------------
cv2.imshow("Step 1 - Gray Image", img_gray)
cv2.imshow("Step 2 - Mask", mask)
cv2.imshow("Step 3 - Mask Inverse", mask_inv)
cv2.imshow("Step 4 - Background Removed", img1_bg)
cv2.imshow("Step 5 - Foreground Extracted", img2_fg)
cv2.imshow("Step 6 - Combined ROI", res)
cv2.imshow("Step 7 - Final Output", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
