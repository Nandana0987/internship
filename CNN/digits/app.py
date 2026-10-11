from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import os

# =====================================
# FLASK APP
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
    "digit_classifier.keras"
)

# =====================================
# PREPROCESS IMAGE
# =====================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    # Convert any color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (28, 28))

    # Auto threshold
    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Normalize
    gray = gray.astype("float32") / 255.0

    # Reshape for CNN
    gray = gray.reshape(1, 28, 28, 1)

    return gray
# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =====================================
# PREDICTION
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

    img = preprocess_image(
        filepath
    )

    prediction = model.predict(
        img,
        verbose=0
    )

    digit = np.argmax(
        prediction
    )

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    return render_template(

        "result.html",

        image_path=filepath,

        digit=digit,

        confidence=confidence

    )

# =====================================
# RUN APP
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )