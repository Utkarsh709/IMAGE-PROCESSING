# capture video from webcam and save into memory
import cv2

cap = cv2.VideoCapture(0)  
# 0 = laptop camera, 1 = external camera
fourcc = cv2.VideoWriter_fourcc(*"XVID")  
# codec
output = cv2.VideoWriter(
    'webcam_video.avi',
    fourcc,
    20.0,
    (640, 480)
)

print("cap", cap)

while cap.isOpened():
    ret, frame = cap.read()  

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #frame=cv2.flip(frame,0)  to flip the video

        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)

        output.write(frame)   # save video

        k = cv2.waitKey(25)
        if k & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
