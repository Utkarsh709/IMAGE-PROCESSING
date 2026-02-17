#take video from user and perfom operation on i

import cv2

cap=cv2.VideoCapture("D:\Image_Processing and Computer Vision\pirates.mp4")
print("cap",cap)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(700,450))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",frame)
    cv2.imshow("gray",gray)
    k=cv2.waitKey(25)
    if k == ord('q') & 0XFF:
        break
  
cap.release()
cv2.destroyAllWindows()    