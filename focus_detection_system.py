import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound
import threading

cv2.namedWindow("Focus Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Focus Tracker", 1200, 800)

def play_sound(freq, duration):
    # Running in a thread prevents the camera feed from freezing
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# ---------------- SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- VARIABLES ----------------
distraction_start_time = None
buffer_start_time = None
focus_stable_start = None
alert_played = False

BUFFER_TIME = 2
ALERT_TIME = 10
RESET_TIME = 3

frame_count = 0
results = [] 

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    current_time = time.time()
    text = "CENTER"
    phone_detected = False
    h, w, _ = frame.shape

    # 1. RUN DETECTION
    # Run YOLO every 3rd frame
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)

    # Face Mesh Detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # 2. PROCESS FACE
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            # Determine direction
            if abs(nose_x - left_eye_x) > abs(nose_x - right_eye_x):
                horizontal = "RIGHT"
            elif abs(nose_x - right_eye_x) > abs(nose_x - left_eye_x):
                horizontal = "LEFT"
            else:
                horizontal = "CENTER"
            
            eye_avg_y = (left_eye_y + right_eye_y) // 2
            vertical = "DOWN" if nose_y > eye_avg_y + 15 else "UP" if nose_y < eye_avg_y - 15 else "CENTER"
            text = f"{vertical} {horizontal}"

    # 3. PROCESS PHONE DETECTION
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. STATE LOGIC
    state = "FOCUSED" if ("CENTER" in text) and not phone_detected else "POSSIBLE_DISTRACTION"

    if state == "POSSIBLE_DISTRACTION":
        if buffer_start_time is None: buffer_start_time = current_time
        state = "DISTRACTED" if (current_time - buffer_start_time > BUFFER_TIME) else "IGNORE"
    else:
        buffer_start_time = None

    # 5. SOUND LOGIC
    if state == "DISTRACTED":
        if distraction_start_time is None:
            distraction_start_time = current_time
            alert_played = False
        if (current_time - distraction_start_time) > ALERT_TIME and not alert_played:
            play_sound(1500, 500)
            alert_played = True
    elif state == "IGNORE":
        if not alert_played:
            play_sound(500, 100)
        alert_played = True
    else:
        alert_played = False

    if state == "FOCUSED":
        if focus_stable_start is None: focus_stable_start = current_time
        if (current_time - focus_stable_start) > RESET_TIME:
            distraction_start_time = None
            alert_played = False
    else:
        focus_stable_start = None

    # 6. DISPLAY
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, state, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Focus Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()