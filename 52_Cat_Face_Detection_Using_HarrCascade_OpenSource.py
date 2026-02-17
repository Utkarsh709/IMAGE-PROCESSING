# Cat Face Detection using Haar Cascade

import cv2
import numpy as np

# -------------------------------
# Load Haar Cascade for cat face
# -------------------------------
cat_cascade = cv2.CascadeClassifier(
    r"D:\Image_Processing and Computer Vision\cascades\haarcascade_frontalcatface_extended.xml"
)

if cat_cascade.empty():
    print("Error: Cat face cascade not loaded")
    exit()

# -------------------------------
# Read image
# -------------------------------
image = cv2.imread(r"D:\Image_Processing and Computer Vision\dogcat.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# -------------------------------
# Convert to grayscale
# -------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Detect cat faces
# parameters: (img, scaleFactor, minNeighbors)
# -------------------------------
faces = cat_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3
)

# -------------------------------
# Draw rectangles
# -------------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

# -------------------------------
# Resize & display
# -------------------------------
image = cv2.resize(image, (800, 700))
cv2.imshow("Cat Face Detected", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
