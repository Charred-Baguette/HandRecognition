from ultralytics import YOLO
import cv2
import numpy as np

class YoloDeployed: 
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.model.fuse()  
        self.model.eval() 

    def predict(self, image, conf=0.25, show=False):
        # Load the image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Image at {image} could not be loaded.")

        # Perform inference
        results = self.model.predict(image, conf=0.25, show=show)

        return results
    def predict_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Make prediction on frame
                results = self.predict(frame, conf=0.25, show=True)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo = YoloDeployed()
    yolo.predict_webcam()