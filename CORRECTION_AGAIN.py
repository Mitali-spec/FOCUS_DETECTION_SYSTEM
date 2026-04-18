import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound
import threading

# --- CONFIGURATION (Adjust these to your liking) ---
GRACE_PERIOD = 4.0    # Absolute silence for 4 seconds when looking away
ALARM_DELAY = 10.0     # Start the annoying alarm after 10 seconds total away
RESET_CONFIRM = 2.0   # Must look back for 2 seconds to "earn" FOCUSED status again

def play_sound(freq, duration):
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# Initialize AI models
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
yolo_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# --- TRACKING VARIABLES ---
current_state = "FOCUSED"
look_away_start = None
focus_restore_start = None
soft_warning_played = False
last_alarm_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    curr_time = time.time()
    h, w, _ = frame.shape
    
    # 1. PERCEPTION: Is the user looking at the screen right now?
    # We run YOLO every 5 frames to keep the app smooth
    phone_detected = False
    if int(curr_time * 10) % 5 == 0:
        results = yolo_model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if yolo_model.names[int(box.cls[0])] == "cell phone":
                    phone_detected = True

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb_frame)
    
    looking_at_screen = False
    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]
        # Nose vs Eyes position logic
        nose = face.landmark[1]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        if left_eye.x < nose.x < right_eye.x:
            looking_at_screen = True

    # 2. THE STATE MACHINE (The "Brain")
    is_behaving = looking_at_screen and not phone_detected

    if is_behaving:
        # If they were distracted, they must look back for RESET_CONFIRM seconds
        if current_state != "FOCUSED":
            if focus_restore_start is None:
                focus_restore_start = curr_time
            
            if (curr_time - focus_restore_start) >= RESET_CONFIRM:
                current_state = "FOCUSED"
                look_away_start = None
                soft_warning_played = False
        else:
            focus_restore_start = None # Already focused
    
    else:
        # User is distracted. Start the timer.
        focus_restore_start = None 
        if look_away_start is None:
            look_away_start = curr_time
        
        time_away = curr_time - look_away_start

        if time_away > ALARM_DELAY:
            current_state = "DISTRACTED"
        elif time_away > GRACE_PERIOD:
            current_state = "POSSIBLE_DISTRACTION"
        else:
            current_state = "GRACE_PERIOD" # Silence here

    # 3. EXECUTION: Sound & UI
    if current_state == "POSSIBLE_DISTRACTION" and not soft_warning_played:
        play_sound(500, 150) # Two soft notification pings
        time.sleep(0.1)
        play_sound(500, 150)
        soft_warning_played = True

    elif current_state == "DISTRACTED":
        # Alarm every 1.5 seconds
        if (curr_time - last_alarm_time) > 1.5:
            play_sound(1200, 400)
            last_alarm_time = curr_time

    # UI Overlay
    msg = f"STATUS: {current_state}"
    color = (0, 255, 0) if current_state == "FOCUSED" else (0, 255, 255) if "PERIOD" in current_state else (0, 0, 255)
    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Anti-Distraction AI", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()