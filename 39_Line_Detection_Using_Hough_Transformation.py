import cv2
import numpy as np

"""
Hough Transform is a popular technique to detect shapes
if those shapes can be represented mathematically.

For line detection, lines are represented as:
Cartesian form : y = mx + c
Polar form     : x*cos(theta) + y*sin(theta) = rho

OpenCV provides:
1. cv2.HoughLines()   -> Standard Hough Transform
2. cv2.HoughLinesP()  -> Probabilistic Hough Transform
"""

# -------------------------------
# Read input image
# -------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\chess.jpg")
img = cv2.resize(img, (400, 400))

# -------------------------------
# Convert image to grayscale
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Edge detection using Canny
# -------------------------------
edges = cv2.Canny(gray, 20, 250)

# -------------------------------
# Apply Probabilistic Hough Transform
# -------------------------------
lines = cv2.HoughLinesP(
    edges,              # edge detected image
    rho=1,              # distance resolution in pixels
    theta=np.pi / 180,  # angle resolution in radians
    threshold=100,      # minimum votes
    minLineLength=100,  # minimum line length
    maxLineGap=100      # maximum allowed gap
)

# -------------------------------
# Draw detected lines
# -------------------------------
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            (100, 200, 125),
            2
        )

# -------------------------------
# Display results
# -------------------------------
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
