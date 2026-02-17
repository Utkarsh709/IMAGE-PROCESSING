#Draw date time and figures on video

import cv2
import datetime

cap=cv2.VideoCapture("D:\Image_Processing and Computer Vision\pirates1.mp4")

#cap=cv2.VideoCapture(0) #do the same thing with the webcam 


print("width:",cap.get(3))  # 3 means width
print("Height:",cap.get(4)) # 4 means height

while(cap.isOpened()):
    ret,frame=cap.read()
    frame=cv2.resize(frame,(900,600))
    if ret==True:
        
        font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        text='  Height:' + str(cap.get(4)) + '  Width:' + str(cap.get(3))
        frame=cv2.putText(frame,text,(10,20),font,1,
                          (0,120,0),1,cv2.LINE_AA)
        
        date_data="Date:  "+str(datetime.datetime.now())
        frame=cv2.putText(frame,date_data,(20,50),font,1,
                          (100,5,255),1,cv2.LINE_AA)
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(27) & 0XFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()        