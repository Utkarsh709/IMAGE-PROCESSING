# ---------------------------------------
# Grayscale Histogram & Equalization
# ---------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------------------
# 1. Read image
# ---------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img = cv2.resize(img, (500, 650))

# ---------------------------------------
# 2. Convert to Grayscale
# ---------------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------
# 3. Plot Grayscale Histogram
# ---------------------------------------
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title("Gray Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

# ---------------------------------------
# 4. Histogram Equalization
# ---------------------------------------
# Works best when image intensity is confined to a narrow range
equ = cv2.equalizeHist(img_gray)

# Stack original and equalized image side-by-side
res = np.hstack((img_gray, equ))

cv2.imshow("Original (Left) | Equalized (Right)", res)

# ---------------------------------------
# 5. Plot Equalized Histogram
# ---------------------------------------
hist1 = cv2.calcHist([equ], [0], None, [256], [0, 256])

plt.plot(hist1)
plt.title("Histogram After Equalization")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

# ---------------------------------------
# 6. Cleanup
# ---------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
