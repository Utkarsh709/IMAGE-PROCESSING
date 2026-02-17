import cv2
import numpy as np

# Load image and convert to grayscale
img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")
img = cv2.resize(img, (600, 600))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dummy callback function for trackbar
def nothing(x):
    pass

# Create window and trackbar
cv2.namedWindow("Canny")
cv2.createTrackbar("Threshold", "Canny", 0, 255, nothing)

while True:
    # Get current trackbar position
    a = cv2.getTrackbarPos("Threshold", "Canny")
    print(a)

    # Apply Canny with dynamic threshold
    res = cv2.Canny(img_gray, a, 255)

    cv2.imshow("Canny", res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # ESC key to exit
        break

cv2.destroyAllWindows()
