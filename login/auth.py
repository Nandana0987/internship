import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection


def recognize_face():
    model = np.load("model/admin_model.npy")

    cap = cv2.VideoCapture(0)

    frame_skip = 0

    with mp_face.FaceDetection(min_detection_confidence=0.7) as detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_skip += 1

            # 🚀 skip frames for speed
            if frame_skip % 10 != 0:
                cv2.imshow("Login", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            if result.detections:

                face = cv2.resize(frame, (80, 80)).flatten()

                dist = np.linalg.norm(face - model)

                # threshold (tune if needed)
                if dist < 3000:
                    cap.release()
                    cv2.destroyAllWindows()
                    return "admin"

            cv2.imshow("Face Login", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return None