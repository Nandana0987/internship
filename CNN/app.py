from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
model = load_model("animal_classifier.h5")

# Load Labels
with open("labels.txt","r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    animal = labels[class_index]

    confidence = round(
        np.max(prediction)*100,
        2
    )

    return render_template(
        "result.html",
        animal=animal,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)