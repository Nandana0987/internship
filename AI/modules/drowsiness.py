import cv2
import numpy as np
import mediapipe as mp
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------
# TIMER
# -------------------------
eyes_closed_start = None


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def ear(landmarks, eye, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye]

    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])

    return (A + B) / (2.0 * C)


def check_drowsiness(frame):
    global eyes_closed_start

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return False

    lm = res.multi_face_landmarks[0].landmark

    left_ear = ear(lm, LEFT_EYE, w, h)
    right_ear = ear(lm, RIGHT_EYE, w, h)

    avg_ear = (left_ear + right_ear) / 2

    current_time = time.time()

    # -------------------------
    # EYES CLOSED CONDITION
    # -------------------------
    if avg_ear < 0.22:
        if eyes_closed_start is None:
            eyes_closed_start = current_time

        from modules.alerts import alert   # add at top if not present

# inside function:
    if eyes_closed_start is not None:
        if current_time - eyes_closed_start >= 3:
            alert("You are drowsy. Please take a rest.", True, True)
            eyes_closed_start = None
            return True
    else:
        eyes_closed_start = None

    return False