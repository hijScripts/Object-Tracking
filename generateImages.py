import cv2
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")

outputDir = "outputImages"
os.makedirs(outputDir, exist_ok = True)

imgCounter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break
    elif k%256 == ord("s"): # "s" key to save the frame
        imgName = os.path.join(outputDir, f"opencv_frame_{imgCounter}.png")
        cv2.imwrite(imgName, frame)
        print(f"{imgName} written!")
        imgCounter += 1

cap.release()
cv2.destroyAllWindows()