# Face Detection using Haar Cascade
import cv2
import numpy as np

# -------------------------------
# Read image
# -------------------------------
image = cv2.imread(r"D:\Image_Processing and Computer Vision\a.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# -------------------------------
# Convert to grayscale
# -------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Load Haar cascade files
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    r"D:\Image_Processing and Computer Vision\cascades\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"D:\Image_Processing and Computer Vision\cascades\haarcascade_eye.xml"
)

# Check if cascades loaded properly
if face_cascade.empty() or eye_cascade.empty():
    print("Error: Haar cascade file not loaded")
    exit()

# -------------------------------
# Detect faces
# parameters: (gray, scaleFactor, minNeighbors)
# -------------------------------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

# -------------------------------
# Draw rectangles on faces & eyes
# -------------------------------
for (x, y, w, h) in faces:
    # Draw face rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (127, 0, 205),
        3
    )

    # Region of Interest for eyes
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    # Detect eyes inside face ROI
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.2,
        minNeighbors=3
    )

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (255, 0, 0),
            2
        )

# -------------------------------
# Resize and show result
# -------------------------------
image = cv2.resize(image, (800, 600))
cv2.imshow("Face Detected", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
