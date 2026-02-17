# Detecting colors using webcam and trackbar

import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def nothing(x):
    pass

cv2.namedWindow("Color Adjustments")

cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)

cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (400, 400))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
 
    # Creating Mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # filter mask with image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Masking", mask)
    cv2.imshow("Result", res)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):   # ESC key
        break

cap.release()
cv2.destroyAllWindows()
