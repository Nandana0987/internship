import pyttsx3
import time
import threading
import winsound
from queue import Queue

engine = pyttsx3.init()
engine.setProperty('rate', 160)

alert_queue = Queue()
last_time = 0
COOLDOWN = 2


# -------------------------
# SPEAK WORKER (SAFE)
# -------------------------
def speaker():
    while True:
        text = alert_queue.get()
        if text is None:
            break

        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

        alert_queue.task_done()


# start background voice thread
threading.Thread(target=speaker, daemon=True).start()


# -------------------------
# BEEP FALLBACK
# -------------------------
def beep():
    try:
        winsound.Beep(1200, 1000)
    except:
        pass


# -------------------------
# MAIN ALERT FUNCTION
# -------------------------
def alert(message, voice=True, sound=True):
    global last_time

    now = time.time()
    if now - last_time < COOLDOWN:
        return

    last_time = now

    if sound:
        beep()

    if voice:
        alert_queue.put(message)