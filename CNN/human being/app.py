import os
import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request
)

from tensorflow.keras.models import load_model

# =====================================
# FLASK CONFIG
# =====================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    "age_gender_ethnicity.keras"
)

# =====================================
# LABELS
# =====================================

gender_labels = {
    0: "Male",
    1: "Female"
}

race_labels = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others"
}

# =====================================
# FACE DETECTOR
# =====================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================
# FACE EXTRACTION
# =====================================

def detect_face(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = img[
        y:y+h,
        x:x+w
    ]

    face = cv2.resize(
    face,
    (96,96)
)
    face = face.astype(
        np.float32
    ) / 255.0

    face = np.expand_dims(
        face,
        axis=0
    )

    return face

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =====================================
# PREDICT
# =====================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "file" not in request.files:

        return render_template(
            "result.html",
            error="No file selected"
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "result.html",
            error="Please select an image"
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    face = detect_face(filepath)

    if face is None:

        return render_template(
            "result.html",
            error="No face detected"
        )

    # ==============================
    # MODEL PREDICTION
    # ==============================

    age_pred, gender_pred, race_pred = model.predict(
        face,
        verbose=0
    )

    # ==============================
    # AGE
    # ==============================

    age = int(
        age_pred[0][0] * 116
    )

    age = max(0, age)

    # ==============================
    # GENDER
    # ==============================

    gender_index = np.argmax(
        gender_pred
    )

    gender = gender_labels[
        gender_index
    ]

    gender_conf = round(
        np.max(gender_pred) * 100,
        2
    )

    # ==============================
    # ETHNICITY
    # ==============================

    race_index = np.argmax(
        race_pred
    )

    race = race_labels[
        race_index
    ]

    race_conf = round(
        np.max(race_pred) * 100,
        2
    )

    # ==============================
    # RESULT
    # ==============================

    return render_template(

        "result.html",

        image_path=filepath,

        age=age,

        gender=gender,

        gender_conf=gender_conf,

        race=race,

        race_conf=race_conf

    )

# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )