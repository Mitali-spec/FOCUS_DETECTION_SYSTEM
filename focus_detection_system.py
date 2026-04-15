import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound  # for sound (Windows)

# ---------------- FACE MESH SETUP ----------------

# mp is mediapipe, solutions is set of tools in mp and face_mesh is tool used to detect face

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ---------------- YOLO SETUP ----------------
model = YOLO("yolov8n.pt")

# ---------------- TRACKING VARIABLES ----------------
look_away_count = 0
last_look_away_time = time.time()
continuous_look_away_start = None

mobile_detect_count = 0
last_mobile_time = time.time()
mobile_start_time = None

alert_played = False
# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera", 1200, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    text = "CENTER"
    phone_detected = False

    # ---------------- FACE DETECTION ----------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            h, w, _ = frame.shape

            # Convert to pixel coordinates
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            left_eye_x = int(left_eye.x * w)
            left_eye_y = int(left_eye.y * h)
            right_eye_x = int(right_eye.x * w)
            right_eye_y = int(right_eye.y * h)

            # Draw points
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)

            # HEAD DIRECTION
            if nose_x < left_eye_x:
                text = "Looking LEFT"
            elif nose_x > right_eye_x:
                text = "Looking RIGHT"
            else:
                text = "CENTER"

            # Vertical
            eye_avg_y = (left_eye_y + right_eye_y) // 2

            if nose_y > eye_avg_y + 15:
                text = "LOOKING DOWN"
            elif nose_y < eye_avg_y - 15:
                text = "LOOKING UP"

            # ---------------- FOCUS LOGIC ----------------
            if text != "CENTER":
                
                last_look_away_time = current_time
                
                if continuous_look_away_start is None:
                    continuous_look_away_start = current_time
                    alert_played = False  # reset when new distraction starts

                elif current_time - continuous_look_away_start > 60:
                     if not alert_played:
                        winsound.Beep(1000, 500)
                        alert_played = True   # prevent repeated beeping

            else:
                # ✅ User came back → RESET EVERYTHING
                continuous_look_away_start = None
                alert_played = False

            # Reset after 2 min focus
            if current_time - last_look_away_time > 120:
                look_away_count = 0

            # Alert if too many distractions
            if look_away_count > 10:
                winsound.Beep(1500, 500)

            # Show text
            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print("Nose:", nose_x, nose_y)

    # ---------------- YOLO PHONE DETECTION ----------------
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---------------- DISPLAY ----------------
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()