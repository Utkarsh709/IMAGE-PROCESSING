import cv2
import numpy as np

# -------------------------------
# 1. Read and preprocess image
# -------------------------------

img = cv2.imread(r"D:\Image_Processing and Computer Vision\hand.jpg")     # Read hand image
img = cv2.resize(img, (600, 700))             # Resize for consistency

# Convert BGR image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise using Median Blur
# Median blur is good for removing salt-and-pepper noise
blur = cv2.medianBlur(img_gray, 9)

# Binary thresholding (invert so hand becomes white)
# White object on black background is best for contour detection
ret, thresh = cv2.threshold(
    blur, 240, 255, cv2.THRESH_BINARY_INV
)

# ------------------------------------
# 2. Find contours
# ------------------------------------
# RETR_EXTERNAL → fetch only outer contour (hand outline)
# CHAIN_APPROX_SIMPLE → compress contour points
cnts, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours:", len(cnts))
print("Hierarchy:\n", hierarchy)

# Draw all contours
cv2.drawContours(img, cnts, -1, (50, 50, 150), 2)

# ------------------------------------
# 3. Draw convex hull for each contour
# ------------------------------------
for c in cnts:
    # Approximate contour to reduce number of points
    epsilon = 0.0001 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Create convex hull
    hull = cv2.convexHull(approx)

    # Draw contour (purple) and hull (green)
    cv2.drawContours(img, [approx], -1, (50, 50, 150), 2)
    cv2.drawContours(img, [hull], -1, (0, 255, 0), 2)

# ------------------------------------
# 4. Convexity Defects (finger gaps)
# ------------------------------------

# Take the largest contour (hand)
c_max = max(cnts, key=cv2.contourArea)

# Convex hull but return indexes (needed for defects)
hull2 = cv2.convexHull(c_max, returnPoints=False)

# Find convexity defects
defects = cv2.convexityDefects(c_max, hull2)

if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(c_max[s][0])
        end   = tuple(c_max[e][0])
        far   = tuple(c_max[f][0])

        # Mark deepest defect point (finger gaps)
        cv2.circle(img, far, 5, (0, 0, 255), -1)

# ------------------------------------
# 5. Extreme Points (Top, Bottom, Left, Right)
# ------------------------------------

# Leftmost point
extLeft = tuple(c_max[c_max[:, :, 0].argmin()][0])
# Rightmost point
extRight = tuple(c_max[c_max[:, :, 0].argmax()][0])
# Topmost point
extTop = tuple(c_max[c_max[:, :, 1].argmin()][0])
# Bottommost point
extBot = tuple(c_max[c_max[:, :, 1].argmax()][0])

# Draw extreme points
cv2.circle(img, extLeft, 8, (255, 0, 255), -1)   # Pink - Left
cv2.circle(img, extRight, 8, (0, 125, 255), -1)  # Brown - Right
cv2.circle(img, extTop, 8, (255, 10, 0), -1)     # Blue - Top
cv2.circle(img, extBot, 8, (19, 152, 152), -1)   # Green - Bottom

# ------------------------------------
# 6. Display results
# ------------------------------------

cv2.imshow("Original with Contours & Hull", img)
cv2.imshow("Gray Image", img_gray)
cv2.imshow("Threshold Image", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
