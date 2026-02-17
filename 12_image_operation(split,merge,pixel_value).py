# Image Operations with pixels and coordinates

import cv2
import numpy as np

# Read an Image
img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img = cv2.resize(img, (600, 400))

print("shape=", img.shape)        # returns (rows, cols, channels)
print("no.of pixels=", img.size)  # total number of pixels
print("datatype=", img.dtype)     # image datatype
print("imagetype=", type(img))    # image type

# Now try to split an image
# split returns 3 channels of our image which is blue, green, red
# print(cv2.split(img))

b, g, r = cv2.split(img)

cv2.imshow("blue", b)
cv2.imshow("green", g)
cv2.imshow("red", r)
cv2.imshow("original", img)

# Now if you want to mix the channels then use merge
mr1 = cv2.merge((r, g, b))
cv2.imshow("rgb", mr1)

mr2 = cv2.merge((g, b, r))
cv2.imshow("gbr", mr2)

cv2.imshow("original again", img)

# access a pixel value by its row and column coordinates
px = img[200, 300]   # store coordinate in variable
print("the pixel of that co-ordinates=", px)

# now try to find selected channel value from coordinate
# We know we have three channel - 0,1,2

# accessing only blue pixel
blue = img[200, 200, 0]
print("the pixel having blue color=", blue)

# for green pass 1
grn = img[200, 300, 1]
print("the pixel having grn color=", grn)

# for red pass 2
red = img[200, 300, 2]
print("the pixel having red color=", red)

cv2.waitKey(0)
cv2.destroyAllWindows()
