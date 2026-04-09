import cv2
import mediapipe as mp


 # mp is mediapipe, solutions is set of tools in mp and face_mesh is tool used to detect face

mp_face_mesh=mp.solutions.face_mesh     

#(blueprint) → FaceMesh
#Object (real thing) → face_mesh
#we are using face_mesh feature of mp which is used to detect face.

face_mesh=mp_face_mesh.FaceMesh()

#cv2.VideoCapture() → function to access camera
#0 IS DEFAULT CAMERA
#“Start the main camera so I can use it”

cap=cv2.VideoCapture(0);

while True:

    #READ FRAME. SHOW ONE FRAME
    ret, frame=cap.read()


     # Convert to RGB (IMPORTANT)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #CONVERTED IMAGE TO RGB
    
    # Process frame
    result = face_mesh.process(rgb_frame) #GAVE CONVERTED IMAGE TO MP


    #multi can detect many faces and face_landmarks means points on face(nose eyse etc)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
        
        #FACE_LANDMARK MEANS POINTS ON FACE.
        #  MP CAN DETECT MANY POINTS ON OUR FACE AND EACH OF THESE POINTS ARE NUMBERS.
        #  HERE 1 MEANS POINT IN NOSE REGION

        nose = face_landmarks.landmark[1]

        h, w, _ = frame.shape   #H IS HEIGHT. W IS WEIGHT. _ MEANS IGNORE THIRD VALUE WITH IS RGB

        # Convert to pixel coordinates
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        # Draw circle on nose
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

        # Print position
        print("Nose:", nose_x, nose_y)



    #SHOW FRAME ON WINDOW / SCREEN

    cv2.imshow("camera",frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
