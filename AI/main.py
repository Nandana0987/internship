import cv2

from modules.drowsiness import check_drowsiness
from modules.yawning import check_yawn
from modules.headpose import check_headpose
from modules.alerts import alert
from modules.utils import save_screenshot, log_event
from fatigue_engine import FatigueEngine
from modules.alerts import alert

# -------------------------
# INIT SYSTEM
# -------------------------
engine = FatigueEngine()

cap = cv2.VideoCapture(0)

print("🚗 AI Driver Safety System Started... Press 'Q' to quit")

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # -------------------------
    # SIGNALS
    # -------------------------
    drowsy = check_drowsiness(frame)
    yawn = check_yawn(frame)
    distracted = check_headpose(frame)

    # -------------------------
    # FATIGUE SCORE
    # -------------------------
    score = engine.update(drowsy, yawn, distracted)
    level = engine.level()

    # -------------------------
    # ALERT SYSTEM (SMART)
    # -------------------------
    if level == "WARNING":
        alert("⚠ Stay alert while driving")

    elif level == "DANGER":
        alert("🚨 Danger! Please stop and rest")
        save_screenshot(frame, "danger")
        log_event("DANGER level fatigue detected")

    # -------------------------
    # DISPLAY STATUS
    # -------------------------
    cv2.putText(frame,
                f"Fatigue Score: {score}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    cv2.putText(frame,
                f"Level: {level}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    # -------------------------
    # DEBUG STATUS
    # -------------------------
    status_text = f"Drowsy:{drowsy}  Yawn:{yawn}  Distracted:{distracted}"
    cv2.putText(frame,
                status_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2)

    # -------------------------
    # SHOW WINDOW
    # -------------------------
    cv2.imshow("AI Driver Safety System", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# CLEANUP
# -------------------------
cap.release()
cv2.destroyAllWindows()

print("System Stopped.")