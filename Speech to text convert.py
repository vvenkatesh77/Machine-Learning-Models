
#speech to text convert

import speech_recognition as sr
r=sr.Recognizer()

with sr.Microphone() as source:
    print("Speak....")
    audio=r.listen(source)

try:
    print("text:",+r.recognize_google(audio))
except Execution:
    print("Its not clear...")

#pip install pipwin
#pipwin install pyaudio

  
