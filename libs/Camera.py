import cv2

class Camera:
    camera = None


    def __init__(self, camIndex):
        self.cammera = cv2.VideoCapture(camIndex)

    def grabFrame(self):
        ret, frame = self.cammera.read()
        return frame