import os
import pickle
import numpy as np
import cv2
import cvzone
import face_recognition
import numpy as np
from datetime import datetime
import math

cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(4, 480)

# imgBackground = cv2.imread('Resources/background.jpeg')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")
print(studentIds)
modeType = 0
counter = 0
id = -1
prev_id = -1
imgStudent = []
taken = False
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    if faceCurFrame:
         counter += 1
         for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # bbox = 10 + x1, 11 + y1, x2 - x1, y2 - y1
         
            if taken:
                img2 = img[x1-60:x2+30,y1-30:y2+100]
                taken = False
                cv2.imwrite('Images/'+str(counter)+'.png',img2)
            # img = cvzone.cornerRect(img, bbox, rt=0)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            perc = face_distance_to_conf(faceDis[matchIndex])
            if perc>0.8:
                id = studentIds[matchIndex]
                if prev_id != id:
                    print(id)
                    prev_id = id

    cv2.imshow("Face Attendance", imgS)
    cv2.waitKey(1)
