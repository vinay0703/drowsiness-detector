#!/usr/bin/env python3
import cv2
import numpy as np
import dlib
###############################################################################

def drawcontours(img,lower,upper):
    """To draw contours when called"""
    for n in range(lower,upper+1):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            next_point=n+1
            if n == upper:
                #To come back to the starting point
                next_point=lower
            x2=landmarks.part(next_point).x
            y2=landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),2)

###############################################################################
facecascade=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
###############################################################################
while True:
    success,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade(img_gray)
    for face in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        landmarks=predictor(img_gray,face)
        #for left eye detection
        drawcontours(img,36,41)
        #for right eye detection
        drawcontours(img,42,47)
        #for external lip detection
        drawcontours(img,48,60)
        #for internal lip detection
        drawcontours(img,61,67)

    cv2.imshow("Cap",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
