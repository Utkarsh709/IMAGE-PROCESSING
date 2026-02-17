# creating Image Border ---->>
# with the help of cv2.copyMakeBorder() function.
# it take parameters like (img, border_width(4-sides), border type,border color)
# border width = top, bottom, right, left

import cv2
import numpy as np

img1 = cv2.imread(r"D:\Image_Processing and Computer Vision\captain_america.jpg")
img1 = cv2.resize(img1, (1000, 600))

# creating image border
brdr = cv2.copyMakeBorder(
    img1,
    10, 10, 5, 5,                 # top, bottom, left, right
    cv2.BORDER_CONSTANT,
    value=[156, 167, 85]           # BGR color
)

cv2.imshow("original image", img1)
cv2.imshow("border image", brdr)

cv2.waitKey(0)
cv2.destroyAllWindows()
