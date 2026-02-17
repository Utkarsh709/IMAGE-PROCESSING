#Buiding a color picker using Trackbars

import cv2
import numpy as np

img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('color Picker')

def cross(x):
    pass

#create trackbar for switch
switch="0:OFF\n1:ON"
cv2.createTrackbar(switch, 'color Picker', 0, 1, cross)

#create trackbar for rgb
cv2.createTrackbar("R",'color Picker',0,255,cross)
cv2.createTrackbar("G",'color Picker',0,255,cross)
cv2.createTrackbar("B",'color Picker',0,255,cross)


while True:
    cv2.imshow("color Picker",img)
    k=cv2.waitKey(1) & 0XFF
    if k == ord('q'):
        break
    
    s=cv2.getTrackbarPos(switch,'color Picker')
    r=cv2.getTrackbarPos("R",'color Picker')
    g=cv2.getTrackbarPos("G",'color Picker')
    b=cv2.getTrackbarPos("B",'color Picker')
    
    if s == 0:
        img[:]=0
    else:
        img[:]=[r,g,b]
        
cv2.destroyAllWindows()        