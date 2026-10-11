from flask import Flask, render_template, request, redirect, session, jsonify, flash
import json
from deepface import DeepFace
import cv2
import os
import sqlite3
import numpy as np 
import PyPDF2
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

score_model = pickle.load(open("score_model.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))
app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ================= CONFIG =================
FACE_DIR = "face_data"
DATABASE = "database.db"
UPLOAD_FOLDER = "uploads"

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= RAG =================
pdf_chunks = []
pdf_vectors = []
pdf_uploaded = False
quiz_store = {}

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS progress(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    module_name TEXT
)
""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quiz_results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        score INTEGER,
        total INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        roll_no TEXT UNIQUE,
        email TEXT UNIQUE,
        course_duration TEXT,
        course_selected TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        role TEXT,
        message TEXT
    )
    """)
    cursor.execute("""
CREATE TABLE IF NOT EXISTS learning_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    source TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

    conn.commit()
    conn.close()

init_db()

# ================= HELPERS =================
def get_student(student_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE id=?", (student_id,))
    student = cursor.fetchone()
    conn.close()
    return student


def save_chat(student_id, role, message):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history(student_id, role, message) VALUES (?,?,?)",
        (student_id, role, message)
    )
    conn.commit()
    conn.close()
def save_learning(student_id, source, content):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO learning_history(student_id, source, content)
        VALUES (?, ?, ?)
    """, (student_id, source, content))

    conn.commit()
    conn.close()
def mark_module_complete(student_id, module_name):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM progress
        WHERE student_id=? AND module_name=?
    """, (student_id, module_name))

    exists = cursor.fetchone()

    if not exists:

        cursor.execute("""
            INSERT INTO progress(student_id,module_name)
            VALUES (?,?)
        """, (student_id,module_name))

        conn.commit()

    conn.close()

# ================= HOME =================

@app.route("/")
def home():
    return render_template("home.html")

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        name = request.form.get("name")
        roll_no = request.form.get("roll_no")
        email = request.form.get("email")
        course_duration = request.form.get("course_duration")
        course_selected = request.form.get("course_selected")

        if not all([name, roll_no, email, course_duration, course_selected]):
            flash("All fields are required", "error")
            return redirect("/register")

        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO students(name, roll_no, email, course_duration, course_selected)
                VALUES (?, ?, ?, ?, ?)
            """, (name, roll_no, email, course_duration, course_selected))

            conn.commit()
            conn.close()

        except sqlite3.IntegrityError:
            flash("Roll No or Email already exists!", "error")
            return redirect("/register")

        # ================= FACE CAPTURE =================
        folder = os.path.join(FACE_DIR, roll_no)
        os.makedirs(folder, exist_ok=True)

        cap = cv2.VideoCapture(0)

        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(f"{folder}/{i}.jpg", frame)
            cv2.imshow("Register", frame)
            cv2.waitKey(300)

        cap.release()
        cv2.destroyAllWindows()

        flash("🎉 Registered successfully!", "success")
        return redirect("/login")

    return render_template("register.html")

# ================= LOGIN =================
@app.route("/login")
def login():
    return render_template("login.html")

# ================= FACE LOGIN =================
@app.route("/detect")
def detect():
    cap = cv2.VideoCapture(0)
    recognized = None

    temp_path = "temp.jpg"
    max_frames = 60
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count > max_frames:
            break

        cv2.imwrite(temp_path, frame)

        for roll in os.listdir(FACE_DIR):
            folder = os.path.join(FACE_DIR, roll)

            for img in os.listdir(folder):
                img_path = os.path.join(folder, img)

                try:
                    result = DeepFace.verify(
                        img1_path=temp_path,
                        img2_path=img_path,
                        enforce_detection=False
                    )

                    if result.get("verified"):
                        recognized = roll
                        break

                except:
                    continue

            if recognized:
                break

        if recognized:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not recognized:
        flash("Face not recognized. Try again or register.", "error")
        return redirect("/login")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE roll_no=?", (recognized,))
    student = cursor.fetchone()
    conn.close()

    if not student:
        flash("Student not found. Please register again.", "error")
        return redirect("/login")

    session["student_id"] = student[0]
    return redirect("/dashboard")

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    flash("👋 Logged out successfully!", "success")
    return redirect("/login")

# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    student_id = session.get("student_id")

    if not student_id:
        flash("Session expired. Please login again.", "error")
        return redirect("/login")

    student = get_student(student_id)

    if not student:
        flash("Student not found", "error")
        return redirect("/login")

    # Calculate progress
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(DISTINCT module_name)
    FROM progress
    WHERE student_id=?
    """,(student_id,))

    completed_modules = cursor.fetchone()[0]

    TOTAL_MODULES = 6

    progress = int((completed_modules / TOTAL_MODULES) * 100)
    return render_template(
        "dashboard.html",
        student=student,
        progress=progress
    )
# ================= AI TUTOR =================
@app.route("/ai-tutor")
def ai_tutor():
    if "student_id" not in session:
        return redirect("/login")
    return render_template("ai_tutor.html")

@app.route("/ask-ai", methods=["POST"])
def ask_ai():

    student_id = session.get("student_id")

    if not student_id:
        return jsonify({"answer": "Session expired"})

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question"})

    # Save user question
    save_chat(student_id, "user", question)
   

    # Get student details
    student = get_student(student_id)

    if not student:
        return jsonify({"answer": "Student not found"})

    course = student[5]      # course_selected column

    # ================= COURSE-SPECIFIC AI TUTOR =================

    if course == "Python Programming":

        system_prompt = """
You are an expert Python Programming tutor.

Teach:
- Variables
- Loops
- Functions
- OOP
- File Handling
- Exception Handling

Always provide Python examples and code snippets.
"""

    elif course == "Data Science Fundamentals":

        system_prompt = """
You are a Data Science tutor.

Teach:
- NumPy
- Pandas
- Statistics
- Data Cleaning
- Data Visualization

Provide practical examples.
"""

    elif course == "Machine Learning":

        system_prompt = """
You are a Machine Learning expert.

Teach:
- Regression
- Classification
- Clustering
- Feature Engineering
- Model Evaluation

Give Python examples using scikit-learn.
"""

    elif course == "Deep Learning":

        system_prompt = """
You are a Deep Learning tutor.

Teach:
- Neural Networks
- CNN
- RNN
- LSTM
- TensorFlow
- PyTorch

Explain concepts step by step.
"""

    elif course == "Database Systems":

        system_prompt = """
You are a Database Systems instructor.

Teach:
- SQL
- Joins
- Normalization
- Transactions
- Indexing

Always provide SQL query examples.
"""

    elif course == "AI & Analytics":

        system_prompt = """
You are an AI and Analytics expert.

Teach:
- Artificial Intelligence
- Data Analytics
- Predictive Analytics
- Business Intelligence
- Data Visualization

Explain with real-world examples.
"""

    else:

        system_prompt = "You are an AI Tutor."

    prompt = f"""
{system_prompt}

Student Question:

{question}

Give detailed explanations with examples.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        answer = response.json().get("response", "No response")

        # Save AI answer
        save_chat(student_id, "bot", answer)
        save_learning(student_id, "AI Tutor", answer)
        mark_module_complete(student_id, "AI Tutor")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": str(e)})
# ================= PDF RAG =================
@app.route("/pdf", methods=["GET", "POST"])
def pdf():
    global pdf_chunks, pdf_vectors, pdf_uploaded

    if "student_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            flash("Select a PDF file", "error")
            return redirect("/pdf")

        path = os.path.join(UPLOAD_FOLDER, "uploaded.pdf")
        file.save(path)

        reader = PyPDF2.PdfReader(path)

        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        words = text.split()
        pdf_chunks = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]

        if not pdf_chunks:
            flash("No text found in PDF", "error")
            return redirect("/pdf")

        pdf_vectors = embed_model.encode(pdf_chunks)
        pdf_uploaded = True

        flash("PDF uploaded successfully", "success")
        return redirect("/pdf")

    return render_template("pdf.html")

# ================= ASK PDF =================
@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    student_id = session.get("student_id")

    if not student_id:
        return jsonify({"answer": "Session expired"})

    if not pdf_uploaded:
        return jsonify({"answer": "Upload PDF first"})

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Enter question"})

    q_vec = embed_model.encode([question])

    scores = cosine_similarity(q_vec, pdf_vectors)[0]
    top = scores.argsort()[-3:][::-1]

    context = " ".join([pdf_chunks[i] for i in top])

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"Context:\n{context}\n\nQuestion: {question}",
                "stream": False
            },
            timeout=60
        )

        answer = response.json().get("response", "")

        save_learning(student_id, "PDF Assistant", answer)
        mark_module_complete(student_id, "PDF Assistant")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": str(e)})

@app.route("/quiz")
def quiz():
    if "student_id" not in session:
        return redirect("/login")

    return render_template("quiz.html")
@app.route("/generate-quiz")
def generate_quiz():

    student_id = session.get("student_id")

    if not student_id:
        return jsonify({"success": False})

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content
        FROM learning_history
        WHERE student_id=?
        ORDER BY id DESC
        LIMIT 20
    """, (student_id,))

    rows = cursor.fetchall()
    conn.close()

    learning_context = "\n".join([row[0] for row in rows])

    student = get_student(student_id)
    course = student[5]

    prompt = f"""
Course:
{course}

Generate 10 multiple-choice questions from ONLY the following learning material.

{learning_context}

Return ONLY valid JSON.

[
{{
"question":"Question",
"options":["A","B","C","D"],
"answer":"Correct Option"
}}
]
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=90
        )

        text = response.json()["response"]

        start = text.find("[")
        end = text.rfind("]") + 1

        questions = json.loads(text[start:end])

        quiz_store[student_id] = questions

        return jsonify({
    "questions": questions
})

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })
@app.route("/get-quiz")
def get_quiz():

    student_id = session.get("student_id")

    if student_id not in quiz_store:
        return jsonify([])

    return jsonify(quiz_store[student_id])

@app.route("/submit-quiz", methods=["POST"])
def submit_quiz():

    student_id = session.get("student_id")

    if student_id not in quiz_store:
        return jsonify({"score": 0})

    answers = request.json["answers"]

    questions = quiz_store[student_id]
    mark_module_complete(student_id, "Quiz")

    score = 0

    for i in range(len(questions)):

        if i < len(answers):

            if answers[i] == questions[i]["answer"]:
                score += 1
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()

                cursor.execute("""
                INSERT INTO quiz_results(student_id,score,total)
                VALUES(?,?,?)
                """, (student_id, score, len(questions)))

                conn.commit()
                conn.close()

    return jsonify({
        "score": score,
        "total": len(questions)
    })
@app.route("/performance")
def performance():

    if "student_id" not in session:
        return redirect("/login")

    student_id = session["student_id"]

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Get average quiz score
    cursor.execute("""
    SELECT AVG(score)
    FROM quiz_results
    WHERE student_id=?
    """, (student_id,))

    row = cursor.fetchone()

    # Handle empty table or no scores
    if row and row[0] is not None:
        avg_score = round(row[0], 2)
    else:
        avg_score = 0

    conn.close()

    # Performance category
    if avg_score >= 8:
        prediction = "Excellent"
        color = "success"

    elif avg_score >= 6:
        prediction = "Good"
        color = "primary"

    elif avg_score >= 4:
        prediction = "Average"
        color = "warning"

    else:
        prediction = "Needs Improvement"
        color = "danger"

    return render_template(
        "performance.html",
        prediction=prediction,
        score=avg_score,
        color=color
    )

@app.route("/score_prediction", methods=["GET", "POST"])
def score_prediction():

    if "student_id" not in session:
        return redirect("/login")

    predicted_score = None

    if request.method == "POST":

        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        resources = int(request.form["resources"])
        extracurricular = int(request.form["extracurricular"])
        motivation = int(request.form["motivation"])
        internet = int(request.form["internet"])
        gender = int(request.form["gender"])
        age = int(request.form["age"])
        learning_style = int(request.form["learning_style"])
        online_courses = int(request.form["online_courses"])
        discussions = int(request.form["discussions"])
        assignment_completion = float(request.form["assignment_completion"])
        edutech = int(request.form["edutech"])
        stress_level = int(request.form["stress_level"])

        predicted_score = score_model.predict([[
            study_hours,
            attendance,
            resources,
            extracurricular,
            motivation,
            internet,
            gender,
            age,
            learning_style,
            online_courses,
            discussions,
            assignment_completion,
            edutech,
            stress_level
        ]])[0]

        predicted_score = round(predicted_score, 2)

    return render_template(
        "score_prediction.html",
        predicted_score=predicted_score
    )
# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)