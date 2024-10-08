import cvzone
import cv2
import numpy as np
import threading


from typing import Tuple

from ultralytics import YOLO

class ObjectDetectionPipeline:
    def __init__(self, modelPath="yolo11n.pt", webcamNum=0, confidenceThreshold=0.5, skipFrame=2):
        self.objectModel = YOLO(modelPath)
        self.webcam: cv2.VideoCapture = cv2.VideoCapture(webcamNum)
        self.confidenceThreshold: float = confidenceThreshold
        self.skipFrame: int = skipFrame
        self.frameCount: int = 0
        self.stopThread: bool = False
        self.frame: np.ndarray = None

    def getParams(self, box) -> Tuple[str, float]:
        """
        Gets the name and confidence value of an object.
        
        This function gets the name of a class and the
        confidence value of the class.
        
        :param object: an object found by YOLO
        :type object: Boxes
        
        :return: Returns the className and confidence value of the object
        :rtype: Tuple[str, float]
        
        :raises ExceptionType: condition
        """

        # Getting num found by yolo and matching to associated name
        classNum = int(box.cls[0])
        className = self.objectModel.names[classNum]

        # Getting confidence value and converting to %
        confidence = float(box.conf[0])
        confidence *= 100

        return className, confidence

    def getCoords(self, box) -> Tuple[int, int, int, int]:
        """
        Gets the top-left and bottom-right coordinates of an
        object.
        
        This function gets the top-left and bottom-right coords as
        (x1, y1) and (x2, y2) respectively.
        
        :param object: an object found by YOLO
        :type object: Boxes
        
        :return: Returns the top-left and bottom-right coords of object
        :rtype: Tuple[int, int, int, int]
        
        :raises ExceptionType: condition
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        return x1, y1, x2, y2

    def captureFrame(self):
        """
        Constantly captures new frames.
        
        This function captures frames from webcam until
        self.stopThread is equal to false.
        
        :raises ExceptionType: condition
        """
        while not self.stopThread:
            frameCaptured, self.frame = self.webcam.read()

            if not frameCaptured:
                print("Failed to capture frame...")
                self.cleanup(self.webcam)
                self.stopThread = True

    def processFrame(self, frame) -> np.ndarray:
        """
        Analyses the frame for objects.
        
        This function analyses the frame for objects and displays
        relevant information.
        
        :param frame: The frame read from the live webcam footage
        :type frame: np.ndarray
        
        :raises ExceptionType: condition
        """

        # Get original frame dimensions
        originalHeight, originalWidth = frame.shape[:2]

        # Resize frame to reduce processing time
        resizedFrame = cv2.resize(frame, (640, 480))

        # Calculate scaling factors between original frame and resized frame
        scaleX = originalWidth / 640
        scaleY = originalHeight / 480

        # Detect objects in frame and then interate over them
        objects = self.objectModel(resizedFrame)
        for object in objects:

            boxes = object.boxes

            for box in boxes:

                # Getting the top-left and bottom-right coords
                x1, y1, x2, y2 = self.getCoords(box)

                # Scale the coordinates back to the original frame size
                x1 = x1 * scaleX
                y1 = y1 * scaleY
                x2 = x2 * scaleX
                y2 = y2 * scaleY

                # Outlining the object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)

                # Getting name & confidence of object
                name, confidence = self.getParams(box)

                # Skip if confidence is below threshold
                if confidence < self.confidenceThreshold:
                   continue

                # Displaying captured values onto frame
                cvzone.putTextRect(frame, f"{name} | {confidence:.2f}% confident.", [x1 + 8, y1 - 12], scale=2)
        
        return frame

    def cleanup(self) -> None:
        """
        Cleans up the program upon completion or exit
        
        This function releases the webcam and destroys all associated
        windows when the program has finished executing.
        
        :param webcam: The users webcam
        :type webcam: cv2. VideoCapture
        
        :raises ExceptionType: condition
        """
        self.webcam.release()
        cv2.DestroyAllWindows()

    def run(self):

        # Start the frame capture thread
        captureThread = threading.Thread(target=self.captureFrame)
        captureThread.start()

        while not self.stopThread:

            processedFrame = self.processFrame(self.frame)
            cv2.imshow("frame", processedFrame)

            keyPress = cv2.waitKey(1)
            if keyPress == 27:
                print("Esc key pressed...")
                self.stopThread = True
                self.cleanup(self.webcam)
        
        captureThread.join()
        
if __name__ == "__main__":
    pipeline = ObjectDetectionPipeline()
    pipeline.run()
