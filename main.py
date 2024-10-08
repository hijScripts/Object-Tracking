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

# Live webcam capture
def initKnownFaces():

    # Path to the folder containing reference images
    image_folder = r"C:\Users\Harry\Desktop\Object Tracking\allowedUsers"

    # Loop through all image files in the folder 
    for image_file in os.listdir(image_folder):
        # Load each image file
        image_path = os.path.join(image_folder, image_file)
        image = face_recognition.load_image_file(image_path)
        
        # Get the face encodings (assuming there's only one face per image)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:

            # Extract the person's name from the filename (e.g., "harry.jpg" -> "harry")
            name = os.path.splitext(image_file)[0]

            if "user1" in name.lower():
                name = "user1"
            elif "user2" in name.lower():
                name = "user2"

            # Store the encoding and the name in the dictionary
            knownFaces[name] = encodings[0]

def detectFace(image):
    faces = faceModel(image)

    for face in faces:
        parameters = face.boxes

        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype("int")
            confidence = box.conf[0]

            faceFrame = image[y1:y2, x1:x2]

            # Convert face frame to RGB for face_recognition
            rgbFace = cv2.cvtColor(faceFrame, cv2.COLOR_BGR2RGB)

            # Compute embeddings of the detected face
            faceEncodings = face_recognition.face_encodings(rgbFace)

            if faceEncodings:
                detectedFaceEncoding = faceEncodings[0]
                
                # Vars to check if an auth face is found and the name of said face.
                foundFace = False
                faceName = ""

                # Compare the detected face with the reference faces
                for name, knownFace in knownFaces.items():
                    results = face_recognition.compare_faces([knownFace], detectedFaceEncoding)

                    if results[0]:
                        foundFace = True
                        faceName = name
                
                if foundFace:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 0), 3)
                    cvzone.putTextRect(image, f"{faceName} | {confidence:.2f}% Confident", [x1 + 8, y1 - 12])
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 255), 3)
                    cvzone.putTextRect(image, f"face | {confidence:.2f}% Confident", [x1 + 8, y1 - 12])

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
