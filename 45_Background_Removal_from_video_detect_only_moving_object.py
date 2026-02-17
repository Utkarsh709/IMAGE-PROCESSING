import numpy as np
import cv2 as cv

"""
Background Subtraction:
-----------------------
Used to extract moving (foreground) objects from a static background.

OpenCV provides multiple background subtraction algorithms.
Here we demonstrate:
1. MOG2 (Gaussian Mixture-based)
2. KNN  (K-Nearest Neighbors-based)
"""

# -------------------------------
# Open video file
# -------------------------------
cap = cv.VideoCapture(r"D:\Image_Processing and Computer Vision\test2.mp4")

if not cap.isOpened():
    print("Error: Video not opened")
    exit()

# -------------------------------
# Create background subtractors
# -------------------------------
algo1 = cv.createBackgroundSubtractorMOG2(detectShadows=True)
algo2 = cv.createBackgroundSubtractorKNN(detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv.resize(frame, (600, 400))

    # Apply background subtraction
    res1 = algo1.apply(frame)
    res2 = algo2.apply(frame)

    # -------------------------------
    # Display results
    # -------------------------------
    cv.imshow("Original Frame", frame)
    cv.imshow("MOG2 Result", res1)
    cv.imshow("KNN Result", res2)

    # Press 'q' or ESC to exit
    key = cv.waitKey(60)
    if key == ord('q') or key == 27:
        break

# -------------------------------
# Release resources
# -------------------------------
cap.release()
cv.destroyAllWindows()
