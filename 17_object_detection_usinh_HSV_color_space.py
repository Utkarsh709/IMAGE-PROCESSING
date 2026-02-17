import cv2
import numpy as np

frame = cv2.imread(r"D:\Image_Processing and Computer Vision\color_balls.jpg")

while True:
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # upper and lower HSV values
    u_v = np.array([130,235,225])
    l_v = np.array([110,50,50])

    # Creating Mask
    mask = cv2.inRange(hsv, l_v, u_v)

    # filter mask with image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == ord('q'):   # ESC key
        break

cv2.destroyAllWindows()
