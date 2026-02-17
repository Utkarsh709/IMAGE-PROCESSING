# Contours and its Functions
# Topics covered:
# 1) Moments
# 2) Contour Approximation
# 3) Convex Hull

import cv2
import numpy as np

# ----------------------------------------
# Load image
# ----------------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\shapes.png")
img = cv2.resize(img, (600, 700))

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
# THRESH_BINARY_INV makes object white and background black
ret, thresh = cv2.threshold(
    img_gray, 200, 255, cv2.THRESH_BINARY_INV
)

# ----------------------------------------
# Find Contours
# findContours(image, retrieval_mode, approximation_method)
# ----------------------------------------
cnts, hier = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

# cnts -> list of contours
# hier -> hierarchy information
print("Number of contours =", len(cnts))
print("Hierarchy =\n", hier)

# ----------------------------------------
# Draw all contours
# ----------------------------------------
cv2.drawContours(img, cnts, -1, (10, 20, 100), 2)

# ----------------------------------------
# Loop over each contour
# ----------------------------------------
area_list = []

for c in cnts:

    # ------------------------------------
    # Moments
    # Image moments are weighted averages of pixel intensities
    # Used to find center (centroid) of the contour
    # ------------------------------------
    M = cv2.moments(c)

    # To avoid division by zero
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Draw contour center
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(
        img,
        "center",
        (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # ------------------------------------
    # Contour Area
    # ------------------------------------
    area = cv2.contourArea(c)
    area_list.append(area)

    # ------------------------------------
    # Contour Approximation
    # Used to reduce number of points
    # epsilon controls approximation accuracy
    # ------------------------------------
    epsilon = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # ------------------------------------
    # Convex Hull
    # Used to get convex shape of the contour
    # ------------------------------------
    hull = cv2.convexHull(approx)

    # ------------------------------------
    # Bounding Rectangle
    # ------------------------------------
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ----------------------------------------
# Display Results
# ----------------------------------------
cv2.imshow("original", img)
cv2.imshow("gray", img_gray)
cv2.imshow("threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
