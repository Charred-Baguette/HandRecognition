import cv2
import os
import mediapipe as mp
import sys
from Camera import Camera

cam = Camera(0)

handsMp = mp.solutions.hands
hands = handsMp.Hands()
mpDraw= mp.solutions.drawing_utils

img_width, img_height = 300, 300


def findPosition(camFrame, handNo=0, draw=False):
    global hands, handsMp, mpDraw
    xList =[]
    yList =[]
    bbox = []
    lmsList=[]


    imgRGB = cv2.cvtColor(camFrame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  
    #if results.multi_hand_landmarks: 
    #    for handLms in results.multi_hand_landmarks:
    #        if draw:
    #            mpDraw.draw_landmarks(camFrame, handLms, handsMp.HAND_CONNECTIONS)
    #        frame = camFrame
    frame = camFrame

        
    
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
        
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            lmsList.append([id, cx, cy])
            if draw:
                cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax
        if draw:
            cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                           (0, 255 , 0) , 2)

    try:
        frame.shape
    except:
        return lmsList, bbox, camFrame
    else:
        return lmsList, bbox, frame
    

while True:
    cameraFrame = cam.grabFrame()
    ogFrame = cameraFrame

    lmsList, bbox, frame2 = findPosition(cameraFrame)

    bbox = list(bbox)

    try:
        for x in range(3):
            if bbox[x] <= 20:
                bbox[x] = 21
    except:
        bbox = 21, 21, 21, 21
    
    print(bbox)


    cv2.imshow('croppedFrame', ogFrame[bbox[1]-20:bbox[3]+20, bbox[0]-20:bbox[2]+20])
    cv2.imshow('frame', cv2.rectangle(frame2, (bbox[0] - 20, bbox[1] - 20),(bbox[2] + 20, bbox[3] + 20), (0, 255 , 0) , 2))
    cv2.waitKey(1)
