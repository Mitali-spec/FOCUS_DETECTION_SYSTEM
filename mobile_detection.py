#IMPORT AND LOAD MODEL

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

#OPEN CAMERA

import cv2

cap=cv2.VideoCapture(0)
cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera", 1200, 800)  # width, height
while True:
    ret , frame =cap.read()
    results=model(frame) #Hey YOLO, look at this image and tell me what objects you see.

    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # class ID
            label = model.names[cls]

            # 🔥 Only detect cell phone
            if label == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("YOLO PHONE DETECTION", frame) #SHOW FRAME ON WINDOW/SCREEN

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release();
cv2.destroyAllWindows()
