import cvzone
import cv2
import face_recognition
import os

from ultralytics import YOLO

# Weights for detecting objects
objectModel = YOLO("yolo11n.pt")
faceModel = YOLO("yolov8n-face.pt") # need to update to v10 trained w faces

# Dict of known faces to compare against
knownFaces = {}

def detectPerson(image):

    # Var to return
    personDetected = False

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

                # Setting person detected to true
                personDetected = True
            
    return personDetected

def main():

    # Initialising known faces dict
    initKnownFaces()

    # Capturing live webcam video
    cap = cv2.VideoCapture(0)

    while True:

        # Getting a frame from footage
        bool, image = cap.read()

        if not bool:
            print("Failed to capture frame... Closing now.")
            break

        # Scanning frame for people
        detected = detectPerson(image)

        # Only detecting faces if people are in frame
        if detected:
            detectFace(image)

            # Displaying captured frame and the people / faces found in it.
            cv2.imshow("frame", image)
            keyPress = cv2.waitKey(1)

            if keyPress%256 == 27:
                print("Escape hit, closing...")
                break
        elif not detected:
            print("No people detected in frame.")

            # Displaying captured frame
            cv2.imshow("frame", image)
            keyPress = cv2.waitKey(1)

            if keyPress%256 == 27:
                print("Escape hit, closing...")
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
