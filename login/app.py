from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import cv2
import base64

app = Flask(__name__)
app.secret_key = "secret"

model = np.load("model/admin_model.npy")


def process_image(image_data):
    # decode base64 image
    img_bytes = base64.b64decode(image_data.split(',')[1])
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (80, 80)).flatten()

    dist = np.linalg.norm(frame - model)

    return dist


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/verify", methods=["POST"])
def verify():
    import base64
    import cv2
    import numpy as np

    data = request.json["image"]

    model = np.load("model/admin_model.npy")

    img_bytes = base64.b64decode(data.split(',')[1])
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (100, 100))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_vector = frame.flatten()

    dist = np.linalg.norm(face_vector - model)

    print("Distance:", dist)

    # ✔ RESET LOGIC (IMPORTANT)
    session.pop("user", None)

    if dist < 4000:
        session["user"] = "admin"
        return {"status": "success"}

    return {"status": "denied"}


@app.route("/admin")
def admin():
    if "user" in session:
        return render_template("admin.html")
    return redirect(url_for("login"))


from flask import session, redirect, url_for

@app.route("/logout")
def logout():
    session.clear()   # clears login session
    return redirect(url_for("login"))  # back to login page


if __name__ == "__main__":
    app.run(debug=True)