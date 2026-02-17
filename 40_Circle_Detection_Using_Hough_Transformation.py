import cv2
import numpy as np

"""
Hough Circle Transform is used to detect circular shapes.
It works by voting in parameter space (x, y, r).

Function used:
cv2.HoughCircles()
"""

# -------------------------------
# Read image
# -------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\col_balls.jpg")
img2 = img.copy()

# -------------------------------
# Convert to grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Reduce noise using Median Blur
# -------------------------------
gray = cv2.medianBlur(gray, 5)

# -------------------------------
# Apply Hough Circle Transform
# Parameters:
# (image, method, dp, minDist, param1, param2, minRadius, maxRadius)
# -------------------------------
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,   # higher threshold for Canny
    param2=30,   # accumulator threshold
    minRadius=0,
    maxRadius=0
)

# -------------------------------
# Draw detected circles
# -------------------------------
if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0, :]:
        # Outer circle
        cv2.circle(img2, (x, y), r, (50, 10, 50), 3)

        # Center of the circle
        cv2.circle(img2, (x, y), 2, (0, 255, 100), -1)

# -------------------------------
# Display results
# -------------------------------
cv2.imshow("Gray Image", gray)
cv2.imshow("Result", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
