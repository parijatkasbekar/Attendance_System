import cv2
import numpy as np
import face_recognition

imgAnurag = face_recognition.load_image_file('/Users/parijatkasbekar/Desktop/Smart-Education-System-main/Base Models/Face Data/Anurag Porel.jpg')
imgAnurag = cv2.cvtColor(imgAnurag,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('/Users/parijatkasbekar/Desktop/Smart-Education-System-main/Base Models/Face Data/Anurag Test1.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgAnurag)[0]
encodeAnurag = face_recognition.face_encodings(imgAnurag)[0]
cv2.rectangle(imgAnurag,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0],faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeAnurag],encodeTest)
faceDis = face_recognition.face_distance([encodeAnurag],encodeTest)
print(results)
cv2.putText(imgTest,f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Anurag Porel',imgAnurag)
cv2.imshow('Anurag Test1',imgTest)
cv2.waitKey(0)