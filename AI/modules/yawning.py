import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

UP = 13
DOWN = 14
LEFT = 61
RIGHT = 291

counter = 0

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def mar(landmarks, w, h):
    def p(i):
        return int(landmarks[i].x*w), int(landmarks[i].y*h)

    return dist(p(UP), p(DOWN)) / dist(p(LEFT), p(RIGHT))

def check_yawn(frame):
    global counter

    h, w = frame.shape[:2]
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return False

    lm = res.multi_face_landmarks[0].landmark
    value = mar(lm, w, h)

    if value > 0.65:
        counter += 1
    else:
        counter = max(0, counter - 1)

    from modules.alerts import alert   # add at top if not present

# inside function:

    if counter >= 3:
        alert("You are yawning. Stay alert while driving.", True, True)
        counter = 0
        return True

    return False