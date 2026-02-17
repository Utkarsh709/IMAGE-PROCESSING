#ROI (Region of Interest)

import numpy as np
import cv2

# read image
img = cv2.imread(r"D:\Image_Processing and Computer Vision\captain_america.jpg")
img = cv2.resize(img, (1422, 800))   # (width, height)

# ROI (Region of Interest)
# Coordinates used:
# (x1, y1) = (603, 56)
# (x2, y2) = (824, 262)
#
# NumPy slicing format:
# img[y1:y2, x1:x2]

roi = img[56:262, 603:824]

cv2.imshow("ROI Image", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
