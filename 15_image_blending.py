# Here we use two important functions cv2.add(), cv2.addWeighted() etc.
# Blending means addition of two images
# If you want to blend two images then both must have same size

import cv2
import numpy as np

img1 = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img1 = cv2.resize(img1, (500, 700))

img2 = cv2.imread(r"D:\Image_Processing and Computer Vision\captain_america.jpg")
img2 = cv2.resize(img2, (500, 700))

cv2.imshow("thor", img1)
cv2.imshow("captain_america", img2)

# Now perform blending
# result = img1 + img2
# numpy addition → modulo operation (overflow wraps around)
# cv2.imshow("result", result)

# recommended to use cv2.add
result1 = cv2.add(img1, img2)   # saturated operation (clipped at 255)
cv2.imshow("result1", result1)

# sum of both the weight w1 + w2 = 1 (max)
# function: cv2.addWeighted(img1, wt1, img2, wt2, gamma_val)

result2 = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
cv2.imshow("result2", result2)

cv2.waitKey(0)
cv2.destroyAllWindows()
