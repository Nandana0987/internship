from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------- MAIN MENU ----------------
def main_menu():
    return {
        "text": "👋 Hi Nandana, welcome to SecureBank Assistant!\nPlease choose an option below:",
        "options": ["💰 Balance", "🏦 Loan Services", "📊 Transactions", "📞 Support"]
    }

# ---------------- BALANCE ----------------
def balance():
    return {
        "text": "💰 Your current account balance is ₹12,45,000.\nWould you like more details?",
        "options": ["📄 Mini Statement", "💸 Transactions", "⬅ Main Menu"]
    }

# ---------------- TRANSACTIONS ----------------
def transactions():
    return {
        "text": "📊 Recent Transactions:\n- ₹500 Debit (ATM)\n- ₹2000 Credit (Salary)\n- ₹300 Debit (UPI)",
        "options": ["💰 Balance", "⬅ Main Menu"]
    }

# ---------------- LOAN MENU ----------------
def loan_menu():
    return {
        "text": "🏦 Loan Services:\nSelect the type of loan you want to explore:",
        "options": ["🏠 Home Loan", "👤 Personal Loan", "🎓 Education Loan", "⬅ Main Menu"]
    }

# ---------------- HOME LOAN ----------------
def home_loan():
    return {
        "text": (
            "🏠 HOME LOAN DETAILS\n\n"
            "💰 Amount: ₹10 Lakhs – ₹5 Crores\n"
            "📊 Interest Rate: 7.2% – 9.5%\n"
            "📅 Tenure: Up to 30 Years\n\n"
            "🏡 PURPOSE:\n"
            "- Buy house\n- Build house\n- Renovation\n\n"
            "📋 ELIGIBILITY:\n"
            "- Age: 21–65 years\n"
            "- Stable income required\n"
            "- Credit score 700+\n\n"
            "✨ BENEFITS:\n"
            "- Low interest rates\n"
            "- Tax benefits (80C)\n"
            "- Long repayment period"
        ),
        "options": ["📝 Apply Home Loan", "📊 EMI Calculator", "📄 Documents Required", "⬅ Loan Menu"]
    }

# ---------------- PERSONAL LOAN ----------------
def personal_loan():
    return {
        "text": (
            "👤 PERSONAL LOAN DETAILS\n\n"
            "💰 Amount: ₹50,000 – ₹25 Lakhs\n"
            "📊 Interest Rate: 10.5% – 18%\n"
            "📅 Tenure: 1 – 5 Years\n\n"
            "🎯 PURPOSE:\n"
            "- Medical expenses\n- Wedding\n- Travel\n- Emergency needs\n\n"
            "⚡ FEATURES:\n"
            "- No collateral required\n"
            "- Fast approval (24–48 hrs)\n"
            "- Minimal documents\n\n"
            "⚠ NOTE:\n"
            "- Higher interest than home loan"
        ),
        "options": ["📝 Apply Personal Loan", "📊 EMI Calculator", "💳 Eligibility Check", "⬅ Loan Menu"]
    }

# ---------------- EDUCATION LOAN ----------------
def education_loan():
    return {
        "text": (
            "🎓 EDUCATION LOAN DETAILS\n\n"
            "💰 Amount: ₹1 Lakh – ₹50 Lakhs\n"
            "📊 Interest Rate: 7% – 10.5%\n"
            "📅 Repayment: After course completion\n\n"
            "📚 PURPOSE:\n"
            "- Higher studies in India\n- Abroad education\n\n"
            "📋 ELIGIBILITY:\n"
            "- Admission proof required\n"
            "- Co-applicant needed\n"
            "- Recognized institution only\n\n"
            "✨ BENEFITS:\n"
            "- Moratorium period\n"
            "- Tax benefits\n"
            "- Flexible repayment"
        ),
        "options": ["📝 Apply Education Loan", "📊 EMI Calculator", "🎓 Course Eligibility", "⬅ Loan Menu"]
    }

# ---------------- SUPPORT ----------------
def support():
    return {
        "text": "📞 Connecting you to customer support...\nPlease wait while we connect you to a banking executive.",
        "options": []
    }
def normalize(msg):
    return msg.lower().replace("⬅", "").replace("🏠", "").replace("👤", "").replace("🎓", "").strip()
# ---------------- ROUTER ----------------
@app.route("/chat", methods=["POST"])
def chat():
    msg = normalize(request.json.get("message", ""))

    # ---------------- MAIN MENU ----------------
    if msg in ["hi", "hello", "start", "menu", "main menu"]:
        return jsonify(main_menu())

    # ---------------- NAVIGATION FIX (IMPORTANT) ----------------
    if "main menu" in msg:
        return jsonify(main_menu())

    if "loan menu" in msg:
        return jsonify(loan_menu())

    # ---------------- LOAN DETAILS ----------------
    if "home loan" in msg:
        return jsonify(home_loan())

    if "personal loan" in msg:
        return jsonify(personal_loan())

    if "education loan" in msg:
        return jsonify(education_loan())

    # ---------------- LOAN ENTRY ----------------
    if "loan" in msg:
        return jsonify(loan_menu())

    # ---------------- BALANCE ----------------
    if "balance" in msg:
        return jsonify(balance())

    # ---------------- TRANSACTIONS ----------------
    if "transaction" in msg:
        return jsonify(transactions())

    # ---------------- SUPPORT ----------------
    if "support" in msg:
        return jsonify(support())

    # ---------------- DEFAULT ----------------
    return jsonify({
        "text": "⚠️ Please choose an option:",
        "options": ["🏠 Home Loan", "👤 Personal Loan", "🎓 Education Loan", "⬅ Main Menu"]
    })
@app.route("/")
def home():
    return render_template("chat.html")
# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)