from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------- MAIN MENU ----------------
def main_menu():
    return {
        "text": "👋 Hi Student, welcome to University Assistant!\nWhat would you like to explore?",
        "options": ["🎓 Courses", "💰 Fees", "📝 Admissions", "📞 Support"]
    }
def clean(msg):
    return (
        msg.lower()
        .replace("⬅", "")
        .replace("🎓", "")
        .replace("💰", "")
        .replace("📝", "")
        .replace("📞", "")
        .strip()
    )
# ---------------- COURSE MENU ----------------
def course_menu():
    return {
        "text": "🎓 Select Course Category:",
        "options": ["UG Courses", "PG Courses", "Professional Courses", "Diploma Courses", "⬅ Main Menu"]
    }

# ---------------- UG COURSES ----------------
def ug_courses():
    return {
        "text": "🎓 UG Courses Available:",
        "options": [
            "BCA", "BSc Computer Science", "BCom", "BA English",
            "BTech", "BSc Nursing", "BPharm", "⬅ Courses"
        ]
    }

# ---------------- PG COURSES ----------------
def pg_courses():
    return {
        "text": "🎓 PG Courses Available:",
        "options": [
            "MCA", "MBA", "MSc Computer Science", "MA English",
            "MSc Physics", "MSc Chemistry", "MSc Maths", "⬅ Courses"
        ]
    }

# ---------------- PROFESSIONAL COURSES ----------------
def professional_courses():
    return {
        "text": "💼 Professional Courses Available:",
        "options": [
            "Data Science", "AI ML", "Web Development", "Cyber Security",
            "⬅ Courses"
        ]
    }

# ---------------- DIPLOMA COURSES ----------------
def diploma_courses():
    return {
        "text": "📜 Diploma Courses Available:",
        "options": [
            "DCA", "Diploma Nursing", "Diploma Engineering",
            "⬅ Courses"
        ]
    }

# ---------------- COURSE DETAILS ----------------
def course_details(course):
    data = {
        "bca": "💻 BCA\nEligibility: +2 Any Stream\nFees: ₹35K–₹75K/year\nDocs: 10th, 12th, TC, ID proof",
        "bsc computer science": "💻 BSc CS\nEligibility: +2 Science\nFees: ₹40K–₹90K/year\nDocs: Marksheets, TC",
        "bcom": "📊 BCom\nEligibility: +2 Any Stream\nFees: ₹25K–₹60K/year",
        "ba english": "📚 BA English\nEligibility: +2 Any Stream\nFees: ₹20K–₹50K/year",
        "btech": "⚙️ BTech\nEligibility: +2 PCM\nFees: ₹80K–₹2.5L/year",
        "bsc nursing": "🧑‍⚕️ BSc Nursing\nEligibility: +2 Biology\nFees: ₹60K–₹1.5L/year",
        "bpharm": "💊 BPharm\nEligibility: +2 Science\nFees: ₹70K–₹2L/year",

        "mca": "💻 MCA\nEligibility: BCA/BSc CS\nFees: ₹50K–₹1.2L/year",
        "mba": "📊 MBA\nEligibility: Any Degree\nFees: ₹80K–₹2.5L/year",
        "msc computer science": "💻 MSc CS\nEligibility: BCA/BSc CS\nFees: ₹60K–₹1.5L/year",
        "ma english": "📚 MA English\nEligibility: BA English\nFees: ₹30K–₹80K/year",
        "msc physics": "🔬 MSc Physics\nEligibility: BSc\nFees: ₹40K–₹1L/year",
        "msc chemistry": "🔬 MSc Chemistry\nEligibility: BSc\nFees: ₹40K–₹1L/year",
        "msc maths": "🔢 MSc Maths\nEligibility: BSc\nFees: ₹40K–₹1L/year",

        "data science": "📈 Data Science\nEligibility: Any Degree\nFees: ₹50K–₹2L",
        "ai ml": "🤖 AI/ML\nEligibility: BSc/BTech/MCA\nFees: ₹80K–₹3L",
        "web development": "🌐 Web Development\nEligibility: Any\nFees: ₹30K–₹1L",
        "cyber security": "🔐 Cyber Security\nEligibility: BCA/BSc/BTech\nFees: ₹60K–₹2.5L",

        "dca": "💻 DCA\nEligibility: +2 Pass\nFees: ₹10K–₹30K",
        "diploma nursing": "🧑‍⚕️ Diploma Nursing\nEligibility: +2 Biology\nFees: ₹30K–₹80K",
        "diploma engineering": "⚙️ Diploma Engineering\nEligibility: SSLC/+2\nFees: ₹20K–₹70K"
    }

    key = course.lower()
    if key in data:
        return {
            "text": data[key],
            "options": ["📝 Apply", "🎓 Courses", "⬅ Main Menu"]
        }

    return {
        "text": "Course not found.",
        "options": ["🎓 Courses", "⬅ Main Menu"]
    }

# ---------------- FEES ----------------
def fees():
    return {
        "text": "💰 Fee Structure:\nUG: ₹20K–₹2.5L/year\nPG: ₹40K–₹3L/year\nDiploma: ₹10K–₹80K",
        "options": ["🎓 Courses", "⬅ Main Menu"]
    }

# ---------------- ADMISSIONS ----------------
def admissions():
    return {
        "text": "📝 Admission Process:\n1. Choose Course\n2. Apply Online\n3. Submit Documents\n4. Merit List\n5. Admission Confirmed",
        "options": ["🎓 Courses", "💰 Fees", "⬅ Main Menu"]
    }

# ---------------- SUPPORT ----------------
def support():
    return {
        "text": "📞 Connecting to University Help Desk...\nPlease wait.",
        "options": []
    }

# ---------------- ROUTER ----------------
@app.route("/chat", methods=["POST"])
def chat():
    msg = clean(request.json.get("message", ""))

    # ---------------- MAIN MENU ----------------
    if msg in ["hi", "hello", "start", "menu", "main menu"]:
        return jsonify(main_menu())

    # ---------------- MAIN NAVIGATION ----------------
    if "main menu" in msg:
        return jsonify(main_menu())

    if "courses" in msg:
        return jsonify(course_menu())

    # ---------------- CATEGORY NAVIGATION ----------------
    if msg == "ug courses":
        return jsonify(ug_courses())

    if msg == "pg courses":
        return jsonify(pg_courses())

    if msg == "professional courses":
        return jsonify(professional_courses())

    if msg == "diploma courses":
        return jsonify(diploma_courses())

    # ---------------- BACK ----------------
    if msg == "courses":
        return jsonify(course_menu())

    # ---------------- FEES / ADMISSION ----------------
    if "fees" in msg:
        return jsonify(fees())

    if "admission" in msg:
        return jsonify(admissions())

    if "support" in msg:
        return jsonify(support())

    # ---------------- COURSE DETAILS ----------------
    return jsonify(course_details(msg))

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("caht1.html")

if __name__ == "__main__":
    app.run(debug=True)