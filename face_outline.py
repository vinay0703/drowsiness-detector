#!/usr/bin/env python3
import cv2
import numpy as np
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

while True:
    success, img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(img_gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img_gray,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Cam",img_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
