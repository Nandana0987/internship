import cv2
import os
from datetime import datetime

# -------------------------
# CREATE REQUIRED FOLDERS
# -------------------------
def ensure_dirs():
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("violations", exist_ok=True)

# -------------------------
# TIMESTAMP
# -------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# -------------------------
# SAVE SCREENSHOT
# -------------------------
def save_screenshot(frame, event="event"):
    ensure_dirs()
    filename = f"violations/{event}_{timestamp()}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# -------------------------
# LOG EVENTS
# -------------------------
def log_event(text):
    ensure_dirs()
    with open("event_log.txt", "a") as f:
        f.write(f"{timestamp()} - {text}\n")