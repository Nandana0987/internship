from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import pickle
import sqlite3
import os

from datetime import datetime

# ==========================
# FLASK CONFIG
# ==========================
app = Flask(__name__)
app.secret_key = "attendance_secret_key"

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================
# MODEL SETTINGS
# ==========================
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 70

# ==========================
# LOAD MODEL
# ==========================
model = tf.keras.models.load_model(
    "employee_model.h5"
)

# ==========================
# LOAD CLASS NAMES
# ==========================
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

print("Loaded Classes:")
print(class_names)

# ==========================
# DATABASE CONNECTION
# ==========================
def get_db_connection():
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    return conn


# ==========================
# DASHBOARD
# ==========================
@app.route("/")
def dashboard():

    conn = get_db_connection()

    today = datetime.now().strftime("%Y-%m-%d")

    # Total Employees
    total_employees = len(class_names)

    # Present Today
    present_today = conn.execute(
        """
        SELECT COUNT(DISTINCT employee_id)
        FROM attendance
        WHERE date = ?
        """,
        (today,)
    ).fetchone()[0]

    absent_today = total_employees - present_today

    total_records = conn.execute(
        """
        SELECT COUNT(*)
        FROM attendance
        """
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        total_employees=total_employees,
        present_today=present_today,
        absent_today=absent_today,
        total_records=total_records
    )


# ==========================
# ATTENDANCE PAGE
# ==========================
@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


# ==========================
# MARK ATTENDANCE
# ==========================
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():

    if "image" not in request.files:
        flash("Please upload an image.")
        return redirect("/attendance")

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.")
        return redirect("/attendance")

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # ======================
    # IMAGE PREPROCESSING
    # ======================
    img = tf.keras.utils.load_img(
        filepath,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    # ======================
    # PREDICTION
    # ======================
    prediction = model.predict(
        img_array,
        verbose=0
    )
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction) * 100)

    print("Predicted:", predicted_class)
    print("Confidence:", confidence)

    confidence = float(
        np.max(prediction) * 100
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    # ======================
    # CONFIDENCE CHECK
    # ======================
    if confidence < CONFIDENCE_THRESHOLD:

        return render_template(
            "result.html",
            success=False,
            message=f"Unknown Employee (Confidence: {confidence:.2f}%)"
        )

    # ======================
    # EXTRACT EMPLOYEE INFO
    # EMP001_Alice
    # ======================
    employee_id = predicted_class.split("_")[0]
    employee_name = predicted_class.split("_")[1]

    current_date = datetime.now().strftime(
        "%Y-%m-%d"
    )

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    conn = get_db_connection()

    # ======================
    # DUPLICATE CHECK
    # ======================
    existing = conn.execute(
        """
        SELECT *
        FROM attendance
        WHERE employee_id=?
        AND date=?
        """,
        (
            employee_id,
            current_date
        )
    ).fetchone()

    if existing:

        conn.close()

        return render_template(
            "result.html",
            success=False,
            message=f"{employee_name} has already marked attendance today."
        )

    # ======================
    # INSERT ATTENDANCE
    # ======================
    conn.execute(
        """
        INSERT INTO attendance
        (
            employee_id,
            employee_name,
            date,
            time,
            confidence
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            employee_id,
            employee_name,
            current_date,
            current_time,
            round(confidence, 2)
        )
    )

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        success=True,
        employee_id=employee_id,
        employee_name=employee_name,
        date=current_date,
        time=current_time,
        confidence=round(confidence, 2)
    )


# ==========================
# ATTENDANCE RECORDS
# ==========================
@app.route("/records")
def records():

    conn = get_db_connection()

    records = conn.execute(
        """
        SELECT *
        FROM attendance
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "records.html",
        records=records
    )


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run(
        debug=True
    )