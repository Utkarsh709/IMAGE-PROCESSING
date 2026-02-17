# Face and Eye Detection using Webcam

import cv2
import numpy as np

# -------------------------------
# Load Haar Cascade files
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    r"D:\Image_Processing and Computer Vision\cascades\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"D:\Image_Processing and Computer Vision\cascades\haarcascade_eye.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Cascade files not loaded")
    exit()

# -------------------------------
# Face & Eye detector function
# -------------------------------
def detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (127, 0, 125),
            3
        )

        # Region of Interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes inside face
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=2
        )

        for (ex, ey, ew, eh) in eyes:
            # Draw eye circle
            cv2.circle(
                roi_color,
                (ex + ew // 2, ey + eh // 2),
                ew // 2,
                (255, 255, 0),
                2
            )

    return img

# -------------------------------
# Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not opened")
    exit()

# -------------------------------
# Webcam loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Apply detector
    output = detector(frame)

    cv2.imshow("Face & Eye Detection", output)

    # Press ENTER or ESC to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
