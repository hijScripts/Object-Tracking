import cvzone
import cv2
import numpy as np

from ultralytics import YOLO

# Weights for detecting objects
objectModel = YOLO("yolo11n.pt")


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
