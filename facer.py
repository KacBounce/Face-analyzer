import cv2 as cv
from deepface import DeepFace
import threading
import time

cap = cv.VideoCapture(0)

con = True
font = cv.FONT_HERSHEY_SIMPLEX
class Analyze(threading.Thread):
    def __init__(self,pre):
        threading.Thread.__init__(self)
        self.pre = pre
    def run(self):
        global img
        global font
        while(con):
            self.pre = DeepFace.analyze(img, enforce_detection=False)    
    def getPre(self):
        return self.pre


def FaceAnalysis_RealTime():
    global img
    global con
    global font
    t2 = Analyze(0)
    start = True
    while con:
        age = []
        _, img = cap.read()
        faceCascade = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(img, 1.1, 4)
        for (x, y, u, v) in faces:
            cv.rectangle(img, (x, y), (x+u, y+v), (0, 255, 0), 2)
            cv.putText(img, "FACE DETECTED", (x, y-30), font,  0.5,
                    (0, 0, 0), 2, cv.LINE_4)
            cv.putText(img, "ANALYZING...", (x, y-15), font,  0.5,
                    (0, 0, 0), 2, cv.LINE_4)
        if(start):
            t2.start()
            start = False
            time.sleep(15)
        prediction = t2.getPre()
        age.append(prediction['age'])
        sex = prediction['gender']
        emotion = prediction['dominant_emotion']
        race = prediction['dominant_race']
        cv.putText(img, ("GENDER : " + sex), (0, 20), font,  0.5,
            (0, 0, 0), 2, cv.LINE_4)
        cv.putText(img, ("EMOTION : " + emotion), (0, 40), font,  0.5,
                (0, 0, 0), 2, cv.LINE_4)
        cv.putText(img, ("RACE : " + race), (0, 60), font,  0.5,
                (0, 0, 0), 2, cv.LINE_4)
        cv.putText(img, ("AGE : " + str(age[0])), (0, 80), font,  0.5,
                   (0, 0, 0), 2, cv.LINE_4)
        cv.imshow("FACE ANALYZER", img)
        if cv.waitKey(1) == ord("q"):
            con = False
            break
    t2.join()

FaceAnalysis_RealTime()
