import cvzone
import cv2
import numpy as np
from typing import Tuple

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes

# Weights for detecting objects
objectModel = YOLO("yolo11n.pt")

def getParams(object: Boxes) -> Tuple[str, float]:
    """
    Purpose:
    
    Desc:
    
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
    Purpose:
    
    Desc:
    
    :param object: an object found by YOLO
    :type object: Boxes
    
    :return: Returns the top-left and bottom-right coords of object
    :rtype: Tuple[int, int, int, int]
    
    :raises ExceptionType: condition
    """
    x1, y1, x2, y2 = map(object.xyxy[0], int)

    return x1, y1, x2, y2


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
