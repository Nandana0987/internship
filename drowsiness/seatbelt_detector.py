import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model(
    "seatbelt_model.h5",
    compile=False
)

# Class names
CLASS_NAMES = [
    "no_seat_belt",
    "seat_belt"
]

IMG_SIZE = 224

def detect_seatbelt(frame):
    """
    Detect seatbelt from a frame

    Returns:
        label
        confidence
        seatbelt_present
    """

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(
        img,
        verbose=0
    )[0][0]

    # Binary classifier
    if prediction > 0.5:

        label = "seat_belt"

        confidence = prediction * 100

        seatbelt_present = True

    else:

        label = "no_seat_belt"

        confidence = (1 - prediction) * 100

        seatbelt_present = False

    return (
        label,
        confidence,
        seatbelt_present
    )


# Standalone Testing
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        label, confidence, seatbelt_present = detect_seatbelt(frame)

        if seatbelt_present:

            color = (0, 255, 0)

            text = f"SEAT BELT ({confidence:.1f}%)"

        else:

            color = (0, 0, 255)

            text = f"NO SEAT BELT ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow(
            "Seatbelt Detection",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()