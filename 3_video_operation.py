#capture video from webcam 
import cv2

cap=cv2.VideoCapture(0) # put 0 bcoz we want to access laptop camera if there external resource is there for camera access then we should put 1 instead of 0 
print("cap",cap)
while cap.isOpened():
    ret,frame=cap.read()  #ret contain boolean value 1 mean video is ongoing and 0 mean video end
    if ret == True:
        frame=cv2.resize(frame,(700,450))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame",frame)
        cv2.imshow("gray",gray)
        k=cv2.waitKey(25)
        if k == ord('q') & 0XFF:
            break
  
cap.release()
cv2.destroyAllWindows()    