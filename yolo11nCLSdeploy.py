from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

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
class YoloDeployed: 
    def __init__(self, model_path='11nCbest.pt'):
        self.model = YOLO(model_path)
        self.model.fuse()  
        self.model.eval() 

    def predict(self, image, conf=0.25, show=False):
        # Load the image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Image at {image} could not be loaded.")

        # Find hand position
        lmsList, bbox, frame = findPosition(image, draw=True)
        
        results = None
        if bbox and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            # Add padding to the crop
            padding = 20
            hand_crop = image[max(0, ymin - padding):min(image.shape[0], ymax + padding),
                            max(0, xmin - padding):min(image.shape[1], xmax + padding)]
            
            if hand_crop.size != 0:  # Check if crop is not empty
                # Perform inference on hand crop
                results = self.model.predict(hand_crop, conf=conf, show=show)

        return results

    def predict_webcam(self):
        cap = cv2.VideoCapture(0)
        
        # Add window configuration
        cv2.namedWindow('Hand Crop', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand Crop', 960, 720)  # 1.5x larger than 640x480
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if success:
                # Find hand position
                lmsList, bbox, frame2 = findPosition(frame, draw=True)
                # Handle bbox coordinates
                if bbox:
                    cv2.imshow('frame', cv2.rectangle(frame2, (bbox[0] - 20, bbox[1] - 20),(bbox[2] + 20, bbox[3] + 20), (0, 255 , 0) , 2))

                    bbox = list(bbox)
                    try:
                        for x in range(3):
                            if bbox[x] <= 20:
                                bbox[x] = 21
                    except:
                        bbox = [21, 21, 21, 21]
                    
                    # Show cropped frame
                    hand_crop = frame[max(0, bbox[1]-20):min(frame.shape[0], bbox[3]+20),
                                    max(0, bbox[0]-20):min(frame.shape[1], bbox[2]+20)]
                    if hand_crop.size != 0:
                        cv2.imshow('Hand Crop', hand_crop)
                
                # Make prediction
                results = self.predict(frame, conf=0.25, show=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo = YoloDeployed()
    yolo.predict_webcam()