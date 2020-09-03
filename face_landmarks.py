#!/usr/bin/env python3
#Shebang line
import cv2
import numpy as np
import dlib
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
cap.set(3,680)
cap.set(4,420)
cap.set(10,100)
while True:
    success,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(img_gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        landmarks=predictor(img_gray,face)
        print(landmarks)

        for n in range(68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            cv2.circle(img,(x,y),3,(0,255,0),cv2.FILLED)

    cv2.imshow("Cap",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
