"""
Lucas–Kanade Optical Flow with Shi–Tomasi Corner Detection
---------------------------------------------------------
Tracks good feature points across video frames and draws
their motion trajectories.
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
# Parameters for Shi–Tomasi corner detection
# -----------------------------------
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# -----------------------------------
# Parameters for Lucas–Kanade optical flow
# -----------------------------------
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Random colors for drawing tracks
color = np.random.randint(0, 255, (100, 3))

# -----------------------------------
# Read first frame
# -----------------------------------
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial Shi–Tomasi corners
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Mask for drawing motion tracks
mask = np.zeros_like(old_frame)

# -----------------------------------
# Tracking loop
# -----------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    # Safety check
    if p1 is None or st is None:
        break

    # Select good points
    good_new = p1[st.flatten() == 1]
    good_old = p0[st.flatten() == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)   # FIXED
        c, d = old.ravel().astype(int)   # FIXED

        mask = cv2.line(
            mask,
            (a, b),
            (c, d),
            color[i % 100].tolist(),
            2
        )
        frame = cv2.circle(
            frame,
            (a, b),
            5,
            color[i % 100].tolist(),
            -1
        )

    # Overlay tracks on frame
    img = cv2.add(frame, mask)
    img = cv2.resize(img, (700, 600))
    cv2.imshow("Lucas–Kanade Optical Flow", img)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

    # Update previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# -----------------------------------
# Cleanup
# -----------------------------------
cap.release()
cv2.destroyAllWindows()
