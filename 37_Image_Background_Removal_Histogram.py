import cv2
import numpy as np

# -------------------------------
# Read original image
# -------------------------------
original_image = cv2.imread(r"D:\Image_Processing and Computer Vision\green.jpg")
original_image = cv2.resize(original_image, (600, 650))
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# -------------------------------
# Read ROI (sample color image)
# -------------------------------
roi = cv2.imread(r"D:\Image_Processing and Computer Vision\g.jpg")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# -------------------------------
# Calculate HSV histogram for ROI
# Using Hue and Saturation channels
# -------------------------------
roi_hist = cv2.calcHist(
    [hsv_roi],
    [0, 1],           # H and S channels
    None,
    [180, 256],       # Number of bins
    [0, 180, 0, 256]  # HSV ranges
)

# Normalize histogram (IMPORTANT)
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# -------------------------------
# Back Projection
# -------------------------------
mask = cv2.calcBackProject(
    [hsv_original],
    [0, 1],
    roi_hist,
    [0, 180, 0, 256],
    scale=1
)

# -------------------------------
# Noise removal using filtering
# -------------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)

# -------------------------------
# Thresholding to get binary mask
# -------------------------------
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

# Convert mask to 3 channels
mask = cv2.merge((mask, mask, mask))

# -------------------------------
# Apply mask to original image
# -------------------------------
result = cv2.bitwise_and(original_image, mask)

# -------------------------------
# Display results
# -------------------------------
cv2.imshow("Original Image", original_image)
cv2.imshow("ROI Image", roi)
cv2.imshow("Back Projection Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
