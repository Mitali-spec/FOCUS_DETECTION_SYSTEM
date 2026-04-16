import cv2
import mediapipe as mp
from ultralytics import YOLO
import time #used to track time
import winsound #used to play sound (beep)
import threading

cv2.namedWindow("Focus Tracker", cv2.WINDOW_NORMAL) #THIS CREATES WINDOW WHERE OUR CAMERA WILL BE SHOWN. FOCUS TRACKER IS NAME OF WINDOW
cv2.resizeWindow("Focus Tracker", 1200, 800) #SETS WINDOW SIZE

#Play a beep sound without stopping the camera”
def play_sound(freq, duration):
    # Running in a thread prevents the camera feed from freezing
    threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()

# ---------------- SETUP ----------------
#mp is the MediaPipe library, solutions is a collection of ready-made modules, and face_mesh is a module used to detect detailed facial landmarks (not just the face).
mp_face_mesh = mp.solutions.face_mesh   #accessing MediaPipe’s face detection tool

face_mesh = mp_face_mesh.FaceMesh()

model = YOLO("yolov8n.pt")  #This loads an object detection model
cap = cv2.VideoCapture(0)   #This starts your camera

#This sets the camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- VARIABLES ----------------
distraction_start_time = None   #NONE MEANS NO VALUE YET
buffer_start_time = None    #Ignore small movements (like blinking or quick look away) FOR SOME TIME
focus_stable_start = None   #“Check if user is focused continuously before resetting system”
soft_alert_played = False    #“Prevent repeating sound again and again” FOR IGNORE
strong_alert_played = False   #FOR DISTRACTION

BUFFER_TIME = 5 #“Wait 5 seconds before deciding the user is distracted”
ALERT_TIME = 20 #“If user is distracted for 20 seconds → give alert”
RESET_TIME = 5  #“User must stay focused for 5 seconds to reset system”

#We use frame_count to skip some frames so that heavy processing like YOLO runs less frequently,
#  improving performance and reducing lag.
frame_count = 0
results = [] 

# ---------------- MAIN LOOP ----------------
while True:
    state = "FOCUSED"
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    current_time = time.time()  #“Get current time (in seconds)”
    text = "DOWN" #“Assume user is looking straight initially”
    phone_detected = False
    h, w, _ = frame.shape   #need height (h) and width (w) mainly for coordinate calculation.

    # 1. RUN DETECTION
    # Run YOLO every 3rd frame. SKIP 2 FRAMES
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)

    # Face Mesh Detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    # 2. PROCESS FACE

     #Did MediaPipe detect any face?
    if result.multi_face_landmarks:

        #“Loop through all detected faces”
        for face_landmarks in result.multi_face_landmarks:


            #THESE ARE MATHS FOR FACE MOVEMENT DETECETION

            nose = face_landmarks.landmark[1]   #GET NOSE, LEFT AND RIGHT EYE
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            #Convert face landmark positions into pixel coordinates

            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            # Determine direction

            #“Check which eye is closer to the nose”

            if abs(nose_x - left_eye_x) > abs(nose_x - right_eye_x):
                horizontal = "RIGHT"
            elif abs(nose_x - right_eye_x) > abs(nose_x - left_eye_x):
                horizontal = "LEFT"
            else:
                horizontal = "CENTER"
            
            eye_avg_y = (left_eye_y + right_eye_y) // 2 #“Find the average Y-position of both eyes”

            vertical = "DOWN" if nose_y > eye_avg_y + 15 else "UP" if nose_y < eye_avg_y - 15 else "CENTER"
            text = f"{vertical} {horizontal}"

    # 3. PROCESS PHONE DETECTION
    for r in results:   #Loop through YOLO results for the frame
        for box in r.boxes: #Loop through each detected object. Each box = one object


            if model.names[int(box.cls[0])] == "cell phone":    #“Check if detected object is a phone”
                phone_detected = True   #“Yes, phone is present in frame”
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    #Draw a box around detected phone

    # 4. STATE LOGIC
    if ("DOWN" in text) and not phone_detected:
        state = "FOCUSED"
        buffer_start_time = None
    else:
        if buffer_start_time is None:
            buffer_start_time = current_time
        elif current_time - buffer_start_time > BUFFER_TIME:
            state = "DISTRACTED"
        else:
            state = "POSSIBLE_DISTRACTION"
        
    # 5. SOUND LOGIC

    #Handle POSSIBLE_DISTRACTION (The "Soft" Warning)
    if state == "POSSIBLE_DISTRACTION":   #User is POSSIBLY distracted 
        if not soft_alert_played:
            play_sound(500, 100) # Soft beep for initial warning
            soft_alert_played = True

    #Handle DISTRACTED (The "Strong" Alert)
    elif state == "DISTRACTED":
        if distraction_start_time is None:
            distraction_start_time = current_time
            strong_alert_played = False
        
        if distraction_start_time is not None and (current_time - distraction_start_time) > ALERT_TIME and not strong_alert_played:
            play_sound(1500, 500) # Loud beep for long-term distraction
            strong_alert_played = True

    #RESET LOGIC (When state is FOCUSED)
    if state == "FOCUSED":
        buffer_start_time = None
        if focus_stable_start is None: focus_stable_start = current_time
        if (current_time - focus_stable_start) > RESET_TIME:
            distraction_start_time = None
            buffer_start_time = None # Reset buffer too
            soft_alert_played = False
            strong_alert_played = False
    


    

    # 6. DISPLAY

    # IT SHOWS TEXT ON WINDOW LIKE FOCUSSED, LOOKING LEFT, ETC
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, state, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Focus Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()