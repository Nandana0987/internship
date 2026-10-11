import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# TIMER FOR DISTRACTION
# -------------------------
distraction_start_time = None


def check_headpose(frame):
    global distraction_start_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    # -------------------------
    # NO FACE DETECTED
    # -------------------------
    if len(faces) == 0:
        if distraction_start_time is None:
            distraction_start_time = current_time

        if current_time - distraction_start_time >= 3:
            return True  # distracted for 3 seconds

        return False

    # -------------------------
    # FACE DETECTED
    # -------------------------
    x, y, w, h = faces[0]

    center_x = x + w / 2
    frame_center = frame.shape[1] / 2

    offset = abs(center_x - frame_center)

    # if looking away
    if offset > frame.shape[1] * 0.25:
        if distraction_start_time is None:
            distraction_start_time = current_time

        if current_time - distraction_start_time >= 3:
            return True
    else:
        # reset when attentive
        distraction_start_time = None

    return False