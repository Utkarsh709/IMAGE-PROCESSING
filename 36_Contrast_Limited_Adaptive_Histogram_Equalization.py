# ---------------------------------------
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------
# Used to enhance local contrast
# Handles noise better than normal histogram equalization
# Works only on grayscale images

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
# 3. Create CLAHE Object
# ---------------------------------------
# clipLimit → contrast limit (higher = more contrast)
# tileGridSize → size of local regions
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE
cl1 = clahe.apply(img_gray)

# ---------------------------------------
# 4. Display Image
# ---------------------------------------
cv2.imshow("CLAHE Output", cl1)

# ---------------------------------------
# 5. Plot CLAHE Histogram
# ---------------------------------------
hist2 = cv2.calcHist([cl1], [0], None, [256], [0, 256])

plt.plot(hist2)
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

# ---------------------------------------
# 6. Cleanup
# ---------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
