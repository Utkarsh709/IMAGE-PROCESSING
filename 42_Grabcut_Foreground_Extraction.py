import cv2
import numpy as np

"""
GrabCut Algorithm:
- Used to separate foreground from background
- Works using Gaussian Mixture Models (GMM)
- Foreground is marked using a rectangle
- Area outside rectangle is treated as background
"""

# -------------------------------
# Read and resize image
# -------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\car.jpg")
img = cv2.resize(img, (800, 800))

# -------------------------------
# Create mask
# -------------------------------
# Mask values:
# 0 -> Background
# 1 -> Foreground
# 2 -> Probable Background
# 3 -> Probable Foreground
mask = np.zeros(img.shape[:2], np.uint8)

# -------------------------------
# Background & Foreground Models
# (Used internally by GrabCut)
# -------------------------------
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# -------------------------------
# Define rectangle for foreground
# Format: (x, y, width, height)
# -------------------------------
rect = (134, 150, 660, 730)

# -------------------------------
# Apply GrabCut
# -------------------------------
cv2.grabCut(
    img,
    mask,
    rect,
    bgdModel,
    fgdModel,
    5,
    cv2.GC_INIT_WITH_RECT
)

# -------------------------------
# Final mask creation
# Keep foreground & probable foreground
# -------------------------------
mask2 = np.where(
    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
    1,
    0
).astype('uint8')

# -------------------------------
# Apply mask to original image
# -------------------------------
result = img * mask2[:, :, np.newaxis]

# -------------------------------
# Display results
# -------------------------------
cv2.imshow("Original Image", img)
cv2.imshow("GrabCut Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
