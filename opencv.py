import cv2

# STEP 1 OPEN CAMERA
cap = cv2.VideoCapture(0) #0 IS DEFAULT CAMERA

while True:
    # READ FRAME
    ret, frame = cap.read() # RET MEANS RETURN TRUE OR FALSE
    #FRAME MEANS A SINGLE PHOTO CAPTURED FROM CAMERA

    # SHOW FRAME
    cv2.imshow("camera", frame) 

    # PRESS Q TO EXIT
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# RELEASE CAMERA AND CLOASE ALL WINDOWS
cap.release()
cv2.destroyAllWindows()
