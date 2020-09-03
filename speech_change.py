#!/usr/bin/env python3
from gtts import gTTS
import playsound
def speak(text,filename):
    """converts text to speech by using google text to speech module(gtts)."""
    tts=gTTS(text=text,slow=False,lang='en')
    tts.save(filename)
    playsound.playsound(filename)
text=input("Enter text that you want to convert to speech: ")
message="Enter the name of speech file that you wish to save."
message+="\nPlease add a prefix .mp3 to save it in mp3 format: "
filename=input(message)
speak(text,filename)
