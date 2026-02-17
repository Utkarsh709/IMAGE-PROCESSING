# ---------------------------------------
# Histogram with Color Image (BGR)
# ---------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------------------
# 1. Read and resize image
# ---------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img = cv2.resize(img, (500, 650))

# ---------------------------------------
# 2. Split BGR channels
# ---------------------------------------
b, g, r = cv2.split(img)

cv2.imshow("Image", img)
cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)

# ---------------------------------------
# 3. Plot histogram for each channel
# ---------------------------------------
plt.figure(figsize=(8, 5))

plt.hist(b.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Blue')
plt.hist(g.ravel(), 256, [0, 256], color='green', alpha=0.5, label='Green')
plt.hist(r.ravel(), 256, [0, 256], color='red', alpha=0.5, label='Red')

plt.title("Color Image Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Number of Pixels")
plt.legend()
plt.show()

# ---------------------------------------
# 4. Wait and cleanup
# ---------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
