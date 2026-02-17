# ---------------------------------------
# Image Histogram: Find, Plot & Analyze
# ---------------------------------------
# Histogram shows intensity distribution
# X-axis → pixel intensity values (0–255)
# Y-axis → number of pixels
# Used to analyze contrast, brightness, exposure

import numpy as np
import cv2
from matplotlib import pyplot as plt

# ---------------------------------------
# 1. Create a blank grayscale image
# ---------------------------------------
img = np.zeros((200, 200), np.uint8)

# Draw rectangles with different intensities
cv2.rectangle(img, (0, 0), (200, 100), 255, -1)   # White region
cv2.rectangle(img, (0, 50), (50, 100), 127, -1)  # Gray region

# ---------------------------------------
# 2. Calculate Histogram using OpenCV
# ---------------------------------------
# Parameters:
# [img] → image list
# [0] → channel (0 for grayscale)
# None → no mask
# [256] → number of bins
# [0,256] → intensity range
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# ---------------------------------------
# 3. Plot Histogram using Matplotlib
# ---------------------------------------
plt.plot(hist)
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Number of Pixels")
plt.show()

# ---------------------------------------
# 4. Display Image
# ---------------------------------------
cv2.imshow("Result Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
