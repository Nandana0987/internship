import cv2
import numpy as np
import uuid

from database import add_person, update_seen

import sqlite3

conn = sqlite3.connect("database/room.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM occupancy")

conn.commit()
conn.close()

print("Old records cleared")


detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

print("Models Loaded Successfully!")

cap = cv2.VideoCapture(0)

known_features = []
known_ids = []

SIMILARITY_THRESHOLD = 0.35

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            feature = recognizer.feature(
                aligned_face
            )

            matched = False

            person_id = None

            for i, known_feature in enumerate(
                known_features
            ):

                score = recognizer.match(
                    feature,
                    known_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > SIMILARITY_THRESHOLD:

                    matched = True
                    person_id = known_ids[i]

                    update_seen(person_id)

                    break

            if not matched:

                person_id = (
                    "PERSON_" +
                    str(uuid.uuid4())[:6]
                )

                known_features.append(feature)
                known_ids.append(person_id)

                add_person(person_id)

                print(
                    "New Person:",
                    person_id
                )

            x, y, bw, bh = map(
                int,
                face[:4]
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+bw, y+bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                person_id,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    cv2.imshow(
        "Smart Room Monitor",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()