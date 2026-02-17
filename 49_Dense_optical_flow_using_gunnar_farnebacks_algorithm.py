"""
Dense Optical Flow using Gunnar Farneback’s Algorithm
-----------------------------------------------------
Shows motion magnitude and direction using HSV color space.

Hue   -> direction of motion
Value -> magnitude (speed) of motion
"""

import cv2
import numpy as np

# -----------------------------------
# Open video
# -----------------------------------
cap = cv2.VideoCapture(r"D:\Image_Processing and Computer Vision\test2.mp4")

if not cap.isOpened():
    print("Error: Video not opened")
    exit()

# -----------------------------------
# Read first frame
# -----------------------------------
ret, frame1 = cap.read()
if not ret:
    print("Error: First frame not read")
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# HSV image for visualization
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # full saturation

# -----------------------------------
# Processing loop
# -----------------------------------
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # -----------------------------------
    # Calculate dense optical flow
    # -----------------------------------
    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # -----------------------------------
    # Convert flow to HSV visualization
    # -----------------------------------
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2   # direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay optical flow on original frame
    output = cv2.add(frame2, rgb)
    output = cv2.resize(output, (700, 430))

    cv2.imshow("Dense Optical Flow (Farneback)", output)

    key = cv2.waitKey(60) & 0xFF
    if key == ord('q'):  # ESC
        break
    elif key == ord('s'):
        cv2.imwrite("optical_frame.png", frame2)
        cv2.imwrite("optical_flow.png", rgb)

    # Update previous frame
    prvs = next_gray

# -----------------------------------
# Cleanup
# -----------------------------------
cap.release()
cv2.destroyAllWindows()
