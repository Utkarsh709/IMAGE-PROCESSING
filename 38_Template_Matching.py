import cv2
import numpy as np

# -------------------------------
# Load target image
# -------------------------------
img = cv2.imread(r"D:\Image_Processing and Computer Vision\avengers.jpg")
img = cv2.resize(img, (800, 600))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Load template
# -------------------------------
template = cv2.imread(r"D:\Image_Processing and Computer Vision\temp.jpg", 0)

best_val = 0
best_loc = None
best_size = None

# -------------------------------
# Multi-scale matching
# -------------------------------
for scale in np.linspace(0.3, 1.5, 30):
    resized = cv2.resize(template, None, fx=scale, fy=scale)

    if resized.shape[0] > gray_img.shape[0] or resized.shape[1] > gray_img.shape[1]:
        continue

    res = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_size = resized.shape[::-1]

print("Best match score:", best_val)

# -------------------------------
# Draw rectangle
# -------------------------------
if best_loc is not None:
    cv2.rectangle(
        img,
        best_loc,
        (best_loc[0] + best_size[0], best_loc[1] + best_size[1]),
        (0, 0, 255),
        2
    )

# -------------------------------
# Display
# -------------------------------
cv2.imshow("Template", template)
cv2.imshow("Detected Result", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
