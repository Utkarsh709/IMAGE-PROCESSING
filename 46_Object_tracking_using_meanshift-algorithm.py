import numpy as np
import cv2 as cv

"""
Object Tracking using MeanShift Algorithm
-----------------------------------------
MeanShift shifts a window towards the region of maximum
density using histogram backprojection.

Steps:
1. Select ROI from first frame
2. Compute HSV histogram of ROI
3. Backproject histogram on each frame
4. Apply MeanShift to update window position
"""

# -------------------------------
# Open video
# -------------------------------
cap = cv.VideoCapture(r"D:\Image_Processing and Computer Vision\test2.mp4")

if not cap.isOpened():
    print("Error: Video not opened")
    exit()

# -------------------------------
# Read first frame
# -------------------------------
ret, frame = cap.read()
if not ret:
    print("Error: First frame not read")
    exit()

# -------------------------------
# Initial tracking window
# (x, y, width, height)
# -------------------------------
x, y, w, h = 580, 30, 80, 150
track_window = (x, y, w, h)

# -------------------------------
# ROI selection
# -------------------------------
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# Mask to ignore low-light colors
mask = cv.inRange(
    hsv_roi,
    np.array((0., 60., 32.)),
    np.array((180., 255., 255.))
)

# -------------------------------
# ROI Histogram
# -------------------------------
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# -------------------------------
# Termination criteria
# (either 10 iterations or move by at least 1 pixel)
# -------------------------------
term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

cv.imshow("ROI", roi)

# -------------------------------
# Tracking loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Backprojection
    dst = cv.calcBackProject(
        [hsv],
        [0],
        roi_hist,
        [0, 180],
        scale=1
    )

    # Apply MeanShift
    ret, track_window = cv.meanShift(dst, track_window, term_criteria)

    # Draw tracking window
    x, y, w, h = track_window
    result = cv.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        3
    )

    # Display
    frame = cv.resize(result, (600, 600))
    cv.imshow("MeanShift Tracking", frame)

    if cv.waitKey(30) & 0xFF == ord('q'):  # ESC key
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv.destroyAllWindows()
