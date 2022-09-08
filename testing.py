import cv2
import numpy as np
import face_recognition

imgRish = face_recognition.load_image_file('ImagesBasic/Ryan Park.jpg')
imgRish = cv2.cvtColor(imgRish, cv2.COLOR_BGR2RGB)

rTest = face_recognition.load_image_file('ImagesBasic/Ryan Iglesias.jpg')
rTest = cv2.cvtColor(rTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRish)[0]
encodeR = face_recognition.face_encodings(imgRish)[0]
cv2.rectangle(imgRish, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,255,0), 2)

faceLocTest = face_recognition.face_locations(rTest)[0]
encodeRt = face_recognition.face_encodings(rTest)[0]
cv2.rectangle(rTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0,255,0), 2)

res = face_recognition.compare_faces([encodeR], encodeRt)
distance = face_recognition.face_distance([encodeR], encodeRt)

print(res, distance)

cv2.putText(rTest, f'{res} {round(distance[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 2)

cv2.imshow('Ryan', imgRish)
cv2.imshow('test', rTest)
cv2.waitKey(0)

