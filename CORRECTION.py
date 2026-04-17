import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound
import threading

#This function plays a beep sound without stopping your program.
def play_sound(freq, duration):
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# ---------------- SETUP ----------------
cv2.namedWindow("Focus Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Focus Tracker", 1200, 800)

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
soft_alert_played = False
strong_alert_played = False

BUFFER_TIME = 5 #“Wait 5 seconds before deciding the user is distracted”
ALERT_TIME = 20 #“If user is distracted for 20 seconds → give alert”
RESET_TIME = 5  #“User must stay focused for 5 seconds to reset system”

frame_count = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    current_time = time.time()
    text = "CENTER CENTER" # Default to centered
    phone_detected = False
    h, w, _ = frame.shape

    # 1. RUN DETECTION
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # 2. FACE MESH
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            horizontal = "RIGHT" if nose_x < left_eye_x else "LEFT" if nose_x > right_eye_x else "CENTER"
            eye_avg_y = (left_eye_y + right_eye_y) // 2
            vertical = "DOWN" if nose_y > eye_avg_y + 15 else "UP" if nose_y < eye_avg_y - 15 else "CENTER"
            text = f"{vertical} {horizontal}"

    # 3. PHONE DETECTION
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. STATE LOGIC
    # Corrected: Focus is looking CENTER, not DOWN
    if ("CENTER" in text) and not phone_detected:
        state = "FOCUSED"
    else:
        if buffer_start_time is None: buffer_start_time = current_time
        state = "DISTRACTED" if (current_time - buffer_start_time > BUFFER_TIME) else "POSSIBLE_DISTRACTION"

    # 5. SOUND LOGIC
    if state == "FOCUSED":
        if focus_stable_start is None: focus_stable_start = current_time
        if (current_time - focus_stable_start) > RESET_TIME:
            distraction_start_time = None
            buffer_start_time = None
            soft_alert_played = 0 # Reset counter
            
    elif state == "POSSIBLE_DISTRACTION":
        # Play exactly 2 low beeps
        if soft_alert_played < 2:
            play_sound(500, 200)
            soft_alert_played += 1
            # Add a small delay between beeps
            time.sleep(0.3) 
            
    elif state == "DISTRACTED":
        # Reset soft alerts so they can play again later
        soft_alert_played = 0 
        
        # Continuous beep: every 1 second, play a high frequency sound
        # We use current_time % 1 to create a recurring trigger
        if int(current_time) % 2 == 0: 
            play_sound(1500, 200)
            
    # 6. DISPLAY
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, state, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Focus Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()