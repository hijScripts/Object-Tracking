import cvzone
import cv2
import numpy as np
from typing import Tuple

from ultralytics import YOLO
from ultralytics import Boxes

# Weights for detecting objects
objectModel = YOLO("yolo11n.pt")

def getParams(object: Boxes) -> Tuple[str, float]:
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
    classNum = int(object.cls[0])
    className = objectModel.names[classNum]

    # Getting confidence value and converting to %
    confidence = float(object.conf[0])
    confidence *= 100

    return className, confidence

def getCoords(object: Boxes) -> Tuple[int, int, int, int]:
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
    x1, y1, x2, y2 = map(int, object.xyxy[0])

    return x1, y1, x2, y2

def analyseFrame(frame: np.ndarray) -> None:
    """
    Analyses the frame for objects.
    
    This function analyses the frame for objects and displays
    relevant information.
    
    :param frame: The frame read from the live webcam footage
    :type frame: np.ndarray
    
    :raises ExceptionType: condition
    """
    # Analysing frame for objects then iterating over them
    objects = objectModel(frame)
    for object in objects:

        # Getting the top-left and bottom-right coords
        x1, y1, x2, y2 = getCoords(object)

        # Outlining the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)

        # Getting name & confidence of object
        name, confidence = getParams(object)

        # Displaying captured values onto frame
        cvzone.putTextRect(frame, f"{name} | {confidence:.2f}% confident.")

def cleanup(webcam: cv2.VideoCapture) -> None:
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

def main():

    # Capturing live webcam video
    cap = cv2.VideoCapture(0)

    while True:

        # Getting a frame from footage
        bool, image = cap.read()

        if not bool:
            print("Failed to capture frame... Closing now.")
            break

        cv2.imshow("frame", image)
        keyPress = cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
