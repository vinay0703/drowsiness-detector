#!/usr/bin/env python3
import os
import cv2
import time
import dlib
import playsound
import numpy as np
from gtts import gTTS
from threading import Thread
from scipy.spatial import distance
###############################################################################
def speak(text):
    """converts text to speech by using google text to speech module(gtts)."""
    tts=gTTS(text=text,slow=False,lang='en')
    file_name="voice.mp3"
    tts.save(file_name)
    playsound.playsound(file_name)

def alarm(path):
    """Alarm for drowsy and yawn detection."""
    playsound.playsound(path)

def drawcontours(img,lower,upper):
    """To draw contour when called and return array of landmarks of eye point"""
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

def getcontour_points(lower,upper,array):
    """Get the contour points x and y and appen them as tuple in eye array"""
    for n in range(lower,upper+1):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            array.append((x,y))
    #to return the eye array
    return array

def get_ear_value(eye):
    """To find the eye_aspect_ratio of an eye"""
    a=distance.euclidean(eye[1],eye[5])
    b=distance.euclidean(eye[2],eye[4])
    c=distance.euclidean(eye[0],eye[3])
    ear=(a+b)/(2.0*c)
    return ear

def get_lip_distance(top_lip,bottom_lip):
    """Returns the lip distance"""
    top_mean=np.mean(top_lip,axis=0)
    bottom_mean=np.mean(bottom_lip,axis=0)
    distance=abs(top_mean[1]-bottom_mean[1])
    return distance

###############################################################################
print("-> Loading the predictor and detector...")
playsound.playsound("predictor.mp3")
facecascade=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("-> Starting Video Stream")
playsound.playsound("cap.mp3")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)
eye_ear_threshold=0.24  #may vary upon distance and quality of camera
                        #adjust it by trail error on line
yawn_threshold=22
count=0
count_threshold=30   #for the frequency of eye closure time and can adjust it
time.sleep(1.0)
#speak("Wake up sir")   #can change the alarm for DROWSINESS from here
alarm_on1=False      #for drowsy alarm
alarm_on2=False      #for yawn alarm
###############################################################################
while True:
    success,img=cap.read()
    #print(img.shape)
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
        #print(eye_ear)  #trail and error method for
                         #adjusting the eye_ear_threshold

        top_lip=[]
        bottom_lip=[]
        #To get the distance between top lip and low lip
        top_lip=getcontour_points(48,54,top_lip)
        bottom_lip=getcontour_points(55,60,bottom_lip)
        lip_distance=get_lip_distance(top_lip,bottom_lip)
        #print(lip_distance)    #trail and error method for
                                #adjusting yawn_threshold

        if eye_ear < eye_ear_threshold:
            count+=1
            if count >= count_threshold:
                if not alarm_on1:
                    alarm_on1=True
                    t=Thread(target=alarm("voice.mp3"),args=("voice.mp3"))
                    t.deamon=True
                    t.start()
                cv2.putText(img,"DROWSINESS ALERT",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            count=0
            alarm_on1=False

        if lip_distance > yawn_threshold:
            if not alarm_on2:
                alarm_on2=True
                t=Thread(target=alarm("yawn.mp3"),args=("yawn.mp3"))
                t.deamon=True
                t.start()
                cv2.putText(img,"YAWN ALERT",(10,30),cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),2)
        else:
            alarm_on2=False

    cv2.putText(img,"EAR: {:.2f}".format(eye_ear),(500,450),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.putText(img,"YAWN: {:.2f}".format(lip_distance),(500,480),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)

    cv2.imshow("Cap",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
