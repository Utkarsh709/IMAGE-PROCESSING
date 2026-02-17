# drawing function in opencv

import numpy as np
import cv2

img = cv2.imread(r"D:\Image_Processing and Computer Vision\thor.png")

if img is None:
    print("Image not found. Check path.")
    exit()

img = cv2.resize(img, (600, 700))


#img=np.zeros([800,800,3],np.uint8)*255  #if we dont want image just use black screen
#img=np.ones([800,800,3],np.uint8)*255  #if we dont want image just use white screen



#line accept 5 parameters (img,start,end,color(BGR),thickness)
img=cv2.line(img,(0,0),(200,200),(43,76,200),5)

#arrowedline accept 5 parameters (img,start,end,color(BGR),thickness)
img=cv2.arrowedLine(img,(0,125),(255,255),(157,176,200),8)

#rectangle accept parameter (img,start_co,end_co,color,thickness)
img=cv2.rectangle(img,(384,10),(510,128),(128,0,255),8)

#circle accept parameter (img,start_co,radius,color,thickness)
img=cv2.circle(img,(447,298),65,(214,255,0),-5)

#puttext accept parameter (img,text,start_co,font,fontsize,color,thickness,linetype)
font=cv2.FONT_HERSHEY_COMPLEX
img=cv2.putText(img, 'THOR', (20,578), font,4,
                (0,145,199),9,cv2.LINE_AA)

#ellipse accept parameter (img,start_cor,(length,height),rotating_point(x,y),ellipse_angle,color,thickness)
img=cv2.ellipse(img,(400,700),(100,50),100,90,270,155,5)



cv2.imshow("res", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
