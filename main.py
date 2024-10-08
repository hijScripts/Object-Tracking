import cvzone
import cv2
import numpy as np
import threading


from typing import Tuple

from ultralytics import YOLO

# Weights for detecting objects
objectModel = YOLO("yolo11n.pt")

class ObjectDetectionPipeline:
    def __init__(self, modelPath="yolo11n.pt", webcam=0, confidenceThreshold=0.5, skipFrame=2):
        self.objectModel = modelPath
        self.webcam = webcam
        self.confidenceThreshold = confidenceThreshold
        self.skipFrame = skipFrame
        self.frameCount = 0
        self.stopThread = False
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
        className = objectModel.names[classNum]

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

    def processFrame(self) -> None:
        """
        Analyses the frame for objects.
        
        This function analyses the frame for objects and displays
        relevant information.
        
        :param frame: The frame read from the live webcam footage
        :type frame: np.ndarray
        
        :raises ExceptionType: condition
        """
        # Analysing frame for objects then iterating over them
        objects = objectModel(self.frame)
        for object in objects:

            boxes = object.boxes

            for box in boxes:

                # Getting the top-left and bottom-right coords
                x1, y1, x2, y2 = self.getCoords(box)

                # Outlining the object
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (50, 50, 255), 3)

                # Getting name & confidence of object
                name, confidence = self.getParams(box)

                # Displaying captured values onto frame
                cvzone.putTextRect(self.frame, f"{name} | {confidence:.2f}% confident.", [x1 + 8, y1 - 12], scale=2)

    def cleanup(self, webcam: cv2.VideoCapture) -> None:
        """
        Cleans up the program upon completion or exit
        
        This function releases the webcam and destroys all associated
        windows when the program has finished executing.
        
        :param webcam: The users webcam
        :type webcam: cv2. VideoCapture
        
        :raises ExceptionType: condition
        """
        webcam.release()
        cv2.DestroyAllWindows()

    def run(self):

        # Capturing live webcam video
        while True:

            # Getting a frame from footage
            bool, frame = cap.read()

            if not bool:
                print("Failed to capture frame... Closing now.")
                cleanup(cap)
                break
            
            # Processing frame
            processFrame(frame)

            # Displaying the frame and exiting loop if 'Esc' is pressed.
            cv2.imshow("frame", frame)
            keyPress = cv2.waitKey(1)

            if keyPress == 27:
                print("Esc key pressed... Closing now.")
                cleanup(cap)
                break
        
if __name__ == "__main__":
    run()
