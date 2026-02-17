import numpy as np
import cv2

"""
Shi–Tomasi Corner Detection
---------------------------
Implemented using cv2.goodFeaturesToTrack()

Compared to Harris:
- More stable
- User-friendly
- Allows control over number of corners and quality
"""

# -------------------------------
# Read image
# -------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\shapes.png")

if img is None:
    print("Error: Image not found")
    exit()

# -------------------------------
# Convert to grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Shi–Tomasi Corner Detection
# Parameters:
# (image, maxCorners, qualityLevel, minDistance)
# -------------------------------
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=5
)

# Convert to integer
corners = np.int64(corners)

# -------------------------------
# Draw corners
# -------------------------------
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# -------------------------------
# Display result
# -------------------------------
cv2.imshow("Shi-Tomasi Corners", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
