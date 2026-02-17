import cv2
import numpy as np

# -------------------------------
# Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

while True:
    # -------------------------------
    # Read frame from camera
    # -------------------------------
    ret, img = cap.read()
    if not ret:
        break

    img2 = img.copy()

    # -------------------------------
    # Convert to grayscale
    # -------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Reduce noise
    # -------------------------------
    gray = cv2.medianBlur(gray, 5)

    # -------------------------------
    # Hough Circle Transform
    # parameters: (image, method, dp, minDist, param1, param2)
    # param1 > param2
    # -------------------------------
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    # -------------------------------
    # Draw detected circles
    # -------------------------------
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for (x, y, r) in circles[0, :]:
            # Outer circle
            cv2.circle(img2, (x, y), r, (50, 10, 50), 3)

            # Center point
            cv2.circle(img2, (x, y), 2, (0, 255, 100), -1)

    # -------------------------------
    # Display result
    # -------------------------------
    cv2.imshow("Result", img2)
    cv2.imshow("Gray", gray)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# -------------------------------
# Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
