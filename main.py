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
    
    >>> name, confidence = getParams(object: Boxes) -> Tuple[str, float]
    name = person, confidence = 87.234324234234%
    """

    # Getting num found by yolo and matching to associated name
    classNum = int(object.cls[0])
    className = objectModel.names[classNum]

    # Getting confidence value and converting to %
    confidence = float(object.conf[0])
    confidence *= 100

    return className, confidence


def detectPerson(image):

    # Finding all objects in frame
    objects = objectModel(image)

    # Going over objects
    for object in objects:

        parameters = object.boxes

        for box in parameters:

            # Getting class number and converting to string
            classDetected = box.cls[0]
            classDetected = int(classDetected)
            classDetectedName = objects[0].names[classDetected]

            # Only displaying persons
            if classDetectedName == "person":
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype("int")
                confidence = box.conf[0]

                cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 255), 3)
                cvzone.putTextRect(image, f"{classDetectedName} | {confidence:.2f}% Confident", [x1 + 8, y1 - 12])

def main():

    # Capturing live webcam video
    cap = cv2.VideoCapture(0)

    while True:

        # Getting a frame from footage
        bool, image = cap.read()

        if not bool:
            print("Failed to capture frame... Closing now.")
            break

        # Scanning frame for people
        detectPerson(image)

        cv2.imshow("frame", image)
        keyPress = cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
