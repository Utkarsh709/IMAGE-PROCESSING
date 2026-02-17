# -----------------------------------------
# Contour Detection using Color Space (HSV)
# -----------------------------------------

import cv2
import numpy as np

# -----------------------------------------
# 1. Open Webcam
# -----------------------------------------
cap = cv2.VideoCapture(0)

# -----------------------------------------
# 2. Dummy function for trackbars
# -----------------------------------------
def nothing(x):
    pass

# -----------------------------------------
# 3. Create Trackbar Window
# -----------------------------------------
cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", 300, 300)

# Threshold trackbar (used after masking)
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

# -----------------------------------------
# 4. HSV Color Range Trackbars
# -----------------------------------------
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 179, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)

cv2.createTrackbar("Upper_H", "Color Adjustments", 179, 179, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

# -----------------------------------------
# 5. Start Processing Frames
# -----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (400, 400))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -----------------------------------------
    # 6. Get HSV Trackbar Values
    # -----------------------------------------
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")

    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # -----------------------------------------
    # 7. Create Mask
    # -----------------------------------------
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply mask on original frame
    filter_img = cv2.bitwise_and(frame, frame, mask=mask)

    # -----------------------------------------
    # 8. Threshold + Morphology
    # -----------------------------------------
    mask_inv = cv2.bitwise_not(mask)

    thresh_val = cv2.getTrackbarPos("Thresh", "Color Adjustments")
    _, thresh = cv2.threshold(mask_inv, thresh_val, 255, cv2.THRESH_BINARY)

    # Remove noise and fill gaps
    dilated = cv2.dilate(thresh, None, iterations=6)

    # -----------------------------------------
    # 9. Find Contours
    # -----------------------------------------
    cnts, hier = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -----------------------------------------
    # 10. Draw Contours & Convex Hull
    # -----------------------------------------
    for c in cnts:
        epsilon = 0.0001 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        hull = cv2.convexHull(approx)

        cv2.drawContours(frame, [approx], -1, (50, 50, 150), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

    # -----------------------------------------
    # 11. Display Results
    # -----------------------------------------
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filter", filter_img)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):  # ESC key
        break

# -----------------------------------------
# 12. Cleanup
# -----------------------------------------
cap.release()
cv2.destroyAllWindows()
