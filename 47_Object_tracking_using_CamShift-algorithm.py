import cv2 as cv
import numpy as np

"""
CamShift Object Tracking
------------------------
CamShift (Continuously Adaptive Mean Shift) improves MeanShift by
automatically adjusting window size and orientation.

Works best when:
- Object color is distinct
- Background is relatively stable
"""

# -----------------------------------
# Open video file
# -----------------------------------
cap = cv.VideoCapture(r"D:\Image_Processing and Computer Vision\test2.mp4")

if not cap.isOpened():
    print("Error: Video not opened")
    exit()

# -----------------------------------
# Read first frame
# -----------------------------------
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

# -----------------------------------
# Initial tracking window (manually set)
# Format: (x, y, width, height)
# -----------------------------------
x, y, w, h = 580, 30, 80, 150
track_window = (x, y, w, h)

# -----------------------------------
# ROI selection
# -----------------------------------
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# Mask low-brightness pixels
mask = cv.inRange(
    hsv_roi,
    np.array((0., 60., 32.)),
    np.array((180., 255., 255.))
)

# -----------------------------------
# ROI histogram
# -----------------------------------
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# -----------------------------------
# Termination criteria
# -----------------------------------
term_criteria = (
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
    10,
    1
)

cv.imshow("Initial ROI", roi)

# -----------------------------------
# Tracking loop
# -----------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Back projection
    dst = cv.calcBackProject(
        [hsv],
        [0],
        roi_hist,
        [0, 180],
        1
    )

    # Apply CamShift
    ret, track_window = cv.CamShift(dst, track_window, term_criteria)

    # Get rotated rectangle points
    pts = cv.boxPoints(ret)
    pts = pts.astype(int)   # FIX for NumPy >= 1.24

    # Draw tracking window
    result = cv.polylines(frame, [pts], True, (0, 255, 0), 3)

    result = cv.resize(result, (600, 600))
    cv.imshow("CamShift Tracking", result)

    if cv.waitKey(30) & 0xFF == ord('q'):  # ESC to exit
        break

# -----------------------------------
# Cleanup
# -----------------------------------
cap.release()
cv.destroyAllWindows()
