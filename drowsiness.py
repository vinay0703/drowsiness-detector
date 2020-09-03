#!/usr/bin/env python3
import time
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
###############################################################################
def drawcontours(img,lower,upper):
    """To draw contours when called and return array of landmarks of eye point"""
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

def getcontour_points(lower,upper,eye_array):
    """Get the contour points x and y and appen them as tuple in eye array"""
    for n in range(lower,upper+1):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            eye_array.append((x,y))
    #to return the eye array
    return eye_array

def get_ear_value(eye):
    """To find the eye_aspect_ratio of an eye"""
    a=distance.euclidean(eye[1],eye[5])
    b=distance.euclidean(eye[2],eye[4])
    c=distance.euclidean(eye[0],eye[3])
    ear=(a+b)/(2.0*c)
    return ear

###############################################################################
facecascade=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
eye_ear_threshold=0.24  #may vary upon distance and quality of camera
                        #adjust it by trail error on line
count=0
time.sleep(1.0)
###############################################################################
while True:
    success,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade(img_gray)
    for face in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        landmarks=predictor(img_gray,face)
        #for left eye detection
        left_eye=drawcontours(img,36,41)
        #for right eye detection
        right_eye=drawcontours(img,42,47)
        #for external lip detection
        drawcontours(img,48,60)
        #for internal lip detection
        #drawcontours(img,61,67)

        left_eye=[]
        right_eye=[]
        #TO get the ear value of left eye
        left_eye=getcontour_points(36,41,left_eye)
        left_eye_ear=get_ear_value(left_eye)
        #To get the ear value of th right eye
        right_eye=getcontour_points(42,47,right_eye)
        right_eye_ear=get_ear_value(right_eye)
        eye_ear=(left_eye_ear+right_eye_ear)/(2.0)
        #print(eye_ear)                 #used for adjusting the camera distance

        if eye_ear < eye_ear_threshold:
            count+=1
            if count >= 30:
                print("Drowsy")
        else:
            count=0

    cv2.imshow("Cap",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
