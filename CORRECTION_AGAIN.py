#NOT AT ALL WORKING

import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound
import threading

def play_sound(freq, duration):
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# ---------------- SETUP ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- VARIABLES ----------------
BUFFER_TIME = 5
RESET_TIME = 5
THRESHOLD = 25 # Increased threshold to prevent flickering

buffer_start_time = None
focus_stable_start = None
last_beep_time = 0
state = "FOCUSED"
state_counter = 0 # To stabilize state changes

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    current_time = time.time()
    
    # 1. DETECTION
    results = model(frame, verbose=False)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    
    direction = "CENTER"
    phone_detected = False

    # 2. FACE MESH LOGIC (With Threshold)
    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks:
            nose = lm.landmark[1]
            left_eye = lm.landmark[33]
            right_eye = lm.landmark[263]
            
            nx, lx, rx = int(nose.x * w), int(left_eye.x * w), int(right_eye.x * w)
            
            if nx < lx - THRESHOLD: direction = "RIGHT"
            elif nx > rx + THRESHOLD: direction = "LEFT"
            else: direction = "CENTER"

    # 3. PHONE DETECTION
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "cell phone":
                phone_detected = True

    # 4. STABILIZED STATE LOGIC
    new_state = "FOCUSED" if (direction == "CENTER" and not phone_detected) else "DISTRACTED"
    
    # Only change state if it persists for 3 frames to stop flickering
    if new_state != state:
        state_counter += 1
        if state_counter > 3:
            state = new_state
            state_counter = 0
    else:
        state_counter = 0

    # 5. NON-BLOCKING SOUND LOGIC
    if state == "DISTRACTED":
        if buffer_start_time is None: buffer_start_time = current_time
        
        # Possible Distraction (0-5 seconds)
        if (current_time - buffer_start_time) < BUFFER_TIME:
            if current_time - last_beep_time > 1.0: # Beep every second
                play_sound(500, 200)
                last_beep_time = current_time
        # Strong Distraction (After 5 seconds)
        else:
            if current_time - last_beep_time > 0.5:
                play_sound(1500, 300)
                last_beep_time = current_time
    else:
        buffer_start_time = None
        last_beep_time = 0

    # 6. DISPLAY
    cv2.putText(frame, f"Look: {direction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"State: {state}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Focus Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()