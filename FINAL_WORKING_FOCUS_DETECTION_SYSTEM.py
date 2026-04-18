import cv2
import mediapipe as mp
from ultralytics import YOLO
import time #USED WHEN YOU WANT TO WORK WITH TIME
import winsound #play sound in your program (Windows only)
import threading    #TO RUN run multiple things at the same time

cv2.namedWindow("Focus Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Focus Tracker", 1200, 800)

# --- CONFIGURATION (Adjust these to your liking) ---
GRACE_PERIOD = 4.0    # When you look away, nothing happens for 4 seconds
ALARM_DELAY = 10.0     # If you keep looking away for 10 seconds, the alarm starts
RESET_CONFIRM = 2.0   # You must look back at the screen for 2 seconds continuously to reset
STRIKE_LIMIT = 2      # 5 soft warnings = Permanent Distraction state

def play_sound(freq, duration):
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# Initialize AI models

#REFINE_LANDMARKS=TRUE MEANS TRACK FACE MORE IN DETAIL ESPECIALLY EYES AND LIPS
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
yolo_model = YOLO("yolov8n.pt")

# THIS IS USED TO INCREASE CAMERA SIZE IN CAMERA WINDOW
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- TRACKING VARIABLES ---
current_state = "FOCUSED"
look_away_start = None  #NONE MEANS NO VALUE IS THERE
focus_restore_start = None
soft_warning_played = False
last_alarm_time = 0
distraction_strike_count = 0  # NEW: Tracks how many times you've been warned

while cap.isOpened():   #RUN THIS LOOP AS LONG AS CAMRA IS OPENED
    ret, frame = cap.read()
    if not ret: break

    curr_time = time.time()
    h, w, _ = frame.shape
    
    # 1. PERCEPTION: Is the user looking at the screen right now?
    # We run YOLO every 5 frames to keep the app smooth

    #THIS PART IS FOR PHONE DETECTION

    phone_detected = False
    if int(curr_time * 10) % 5 == 0:    #THIS MEANS Run this block roughly every 0.5 seconds BECAUSE camera loop runs very fast (like 30–60 frames/sec)
        results = yolo_model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                if yolo_model.names[int(box.cls[0])] == "cell phone":
                    phone_detected = True

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb_frame)
    
    #This code checks if a detected face is looking at the screen using nose and eye positions.

    looking_at_screen = False
    if res.multi_face_landmarks:
        face = res.multi_face_landmarks[0]
        # Nose vs Eyes position logic
        nose = face.landmark[1]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        #If the nose is between both eyes horizontally → face is looking forward (toward screen)

        if left_eye.x < nose.x < right_eye.x:
            looking_at_screen = True

    # 2. THE STATE MACHINE (The "Brain")

    #If the user looks back at the screen without using a phone and stays focused for a few seconds, mark them as focused again and reset warnings.

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
                distraction_strike_count = 0 # NEW: Reset strikes once you prove you are focused
        
        else:
            focus_restore_start = None 
            look_away_start = None # Reset this so timer starts fresh next look-away
        
    else:
            focus_restore_start = None # Already focused
            if look_away_start is None:
                look_away_start = curr_time
        
            time_away = curr_time - look_away_start

            # LOGIC: Either too much time has passed, OR you've had too many warnings
            if time_away > ALARM_DELAY or distraction_strike_count >= STRIKE_LIMIT:
                current_state = "DISTRACTED"
            elif time_away > GRACE_PERIOD:
                current_state = "POSSIBLE_DISTRACTION"
            else:
                current_state = "GRACE_PERIOD"


    # 3. EXECUTION: Sound & UI

    if current_state == "POSSIBLE_DISTRACTION" and not soft_warning_played:
        play_sound(500, 150) # Two soft notification pings
        time.sleep(0.1)
        play_sound(500, 150)
        soft_warning_played = True
        distraction_strike_count += 1  # NEW: Add a strike for this warning


    elif current_state == "DISTRACTED":
        # Alarm every 1.5 seconds
        if (curr_time - last_alarm_time) > 1.5:
            play_sound(1200, 400)
            last_alarm_time = curr_time

   # UI Overlay
    msg = f"STATUS: {current_state}"
    strikes_msg = f"STRIKES: {distraction_strike_count}/{STRIKE_LIMIT}"
    color = (0, 255, 0) if current_state == "FOCUSED" else (0, 255, 255) if "PERIOD" in current_state else (0, 0, 255)
    
    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, strikes_msg, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Anti-Distraction AI", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()