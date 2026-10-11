import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
import os
import winsound
from datetime import datetime
from seatbelt_detector import detect_seatbelt

# ==============================
# FOLDERS
# ==============================
os.makedirs("recordings", exist_ok=True)
os.makedirs("violations", exist_ok=True)

# ==============================
# VIDEO RECORDING
# ==============================
record_name = f"recordings/record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

cap = cv2.VideoCapture(0)
w = int(cap.get(3))
h = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(record_name, fourcc, 20.0, (w, h))

# ==============================
# VOICE ENGINE
# ==============================
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

voice_lock = threading.Lock()

def speak(text):
    with voice_lock:
        engine.say(text)
        engine.runAndWait()

# ==============================
# ALERT SYSTEM
# ==============================
alert_lock = threading.Lock()
current_alert = None
last_alert_time = 0
ALERT_COOLDOWN = 3

ALERT_PRIORITY = {
    "seatbelt": 1,
    "yawn": 2,
    "distraction": 3,
    "drowsy": 4
}

def clear_alert():
    global current_alert
    current_alert = None

def trigger_alert(alert_type, message):
    global current_alert, last_alert_time

    with alert_lock:

        if time.time() - last_alert_time < ALERT_COOLDOWN:
            return

        current_alert = alert_type
        last_alert_time = time.time()

        # 🔊 BEEP only for critical alerts
        if alert_type in ["drowsy", "yawn", "distraction"]:
            threading.Thread(
                target=lambda: winsound.Beep(2000, 3000),
                daemon=True
            ).start()

        # 🗣️ VOICE
        threading.Thread(
            target=speak,
            args=(message,),
            daemon=True
        ).start()

        threading.Timer(2, clear_alert).start()

# ==============================
# SAVE SYSTEM
# ==============================
last_save_time = {
    "seatbelt": 0,
    "drowsy": 0,
    "yawning": 0,
    "distraction": 0
}

SAVE_COOLDOWN = 10

def save_violation(frame, name):
    now = time.time()

    if now - last_save_time[name] < SAVE_COOLDOWN:
        return

    filename = f"violations/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)

    last_save_time[name] = now

def log_event(event):
    with open("event_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {event}\n")

# ==============================
# MEDIAPIPE
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(lm, eye):
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def mouth_aspect_ratio(lm):
    return euclidean(lm[13], lm[14]) / euclidean(lm[78], lm[308])

# ==============================
# VARIABLES
# ==============================
eyes_closed_start = None

yawn_state = "CLOSED"
yawn_count = 0
last_yawn_time = 0
YAWN_COOLDOWN = 2

looking_away_start = None
last_seatbelt_alert = 0

# ==============================
# MAIN LOOP
# ==============================
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ================= FACE =================
    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            h, w, _ = frame.shape
            lm = [(int(p.x * w), int(p.y * h)) for p in face.landmark]

            # ================= EAR =================
            ear = (eye_aspect_ratio(lm, LEFT_EYE) +
                   eye_aspect_ratio(lm, RIGHT_EYE)) / 2

            cv2.putText(frame, f"EAR:{ear:.2f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            if ear < 0.20:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                elif time.time() - eyes_closed_start >= 3:
                    trigger_alert("drowsy",
                                  "You are sleepy. Please take some rest.")
                    save_violation(frame, "drowsy")
                    log_event("Drowsiness detected")
            else:
                eyes_closed_start = None

            # ================= MAR =================
            mar = mouth_aspect_ratio(lm)

            cv2.putText(frame, f"MAR:{mar:.2f}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

            if yawn_state == "CLOSED":
                if mar > 0.60:
                    yawn_state = "OPEN"
            else:
                if mar < 0.50:
                    if time.time() - last_yawn_time > YAWN_COOLDOWN:
                        yawn_count += 1
                        last_yawn_time = time.time()
                    yawn_state = "CLOSED"

            if yawn_count >= 3:
                cv2.putText(frame, "EXCESSIVE YAWNING",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                trigger_alert("yawn",
                              "You are yawning frequently. Please take some rest.")
                save_violation(frame, "yawning")
                log_event("Yawning detected")

                yawn_count = 0

            # ================= DISTRACTION =================
            nose = lm[1]
            left_face = lm[234]
            right_face = lm[454]

            center = (left_face[0] + right_face[0]) / 2
            offset = nose[0] - center

            if abs(offset) > 20:
                if looking_away_start is None:
                    looking_away_start = time.time()

                elif time.time() - looking_away_start >= 3:
                    trigger_alert("distraction",
                                  "Focus on the road. You are not looking straight.")
                    save_violation(frame, "distraction")
                    log_event("Distracted driving")
            else:
                looking_away_start = None

    # ================= SEATBELT =================
    label, conf, seatbelt = detect_seatbelt(frame)

    if not seatbelt:

        if time.time() - last_seatbelt_alert > 8:
            trigger_alert("seatbelt",
                          "Please wear your seat belt.")
            last_seatbelt_alert = time.time()

        save_violation(frame, "seatbelt")
        log_event("No seatbelt detected")

        text = f"No Seat Belt {conf:.1f}%"
        color = (0, 0, 255)
    else:
        text = f"Seat Belt {conf:.1f}%"
        color = (0, 255, 0)

    cv2.putText(frame, text, (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ================= TIME =================
    cv2.putText(frame,
                datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    # ================= SAVE VIDEO =================
    out.write(frame)

    # ================= SHOW =================
    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
out.release()
cv2.destroyAllWindows()