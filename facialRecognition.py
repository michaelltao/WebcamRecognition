import cv2
import numpy as np
import face_recognition
import os

path = 'Database'
images = []
names = []
myList = os.listdir(path)

for i in myList:
    curImage = cv2.imread(f'{path}/{i}')
    images.append(curImage)
    names.append(os.path.splitext(i)[0])

def encode(imgs):
    encodings = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        encodings.append(enc)
    return encodings

encodingList = encode(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    curFaces = face_recognition.face_locations(imgS)
    curEncodings = face_recognition.face_encodings(imgS, curFaces)

    for eF, loc in zip(curEncodings, curFaces):
        matches = face_recognition.compare_faces(encodingList, eF)
        faceDis = face_recognition.face_distance(encodingList, eF)
        matchIdx = np.argmin(faceDis)

        if matches[matchIdx]:
            name = names[matchIdx].upper()
            y1, x2, y2, x1 = loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)















