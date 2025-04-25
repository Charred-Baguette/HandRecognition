import cv2

class Camera:
    camera = None

    def __init__(self, camIndex):
        self.camera = cv2.VideoCapture(camIndex) 
        
    def grabFrame(self):
        ret, frame = self.camera.read()
        return frame if ret else None

    def release(self):
        if self.camera is not None:
            self.camera.release()